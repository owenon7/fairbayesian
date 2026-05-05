import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from scipy.stats import beta, binom
from scipy.special import betainc

# Parameters
alpha = 0.000001/2
intervals = 1000
x_values = np.linspace(1/(2*intervals), 1 - 1/(2*intervals), intervals)
hbaseline = 100
a0 = 1
b0 = 1

def DataPrep(data, a0, b0, x_values):
    print('DataPrep')
    # Summarise data into d nodes each with a Count and Target Count
    combos = pd.merge(pd.DataFrame(data.loc[:, data.columns != 'target'].value_counts()).reset_index(drop=False).rename({'count':'Count'}, axis=1), 
                    pd.DataFrame(data[data['target'] == 1].loc[:,data.columns != 'target'].value_counts()).reset_index(drop=False).rename({'count':'Target'}, axis=1), 
                    how='outer').fillna(0)

    # Get variables needed for inferring the beta distribution of each d node
    combos['ad'] = combos['Target'] + a0
    combos['bd'] = combos['Count'] - combos['Target'] + b0

    # Calc beta dists – vectorised over unique (ad, bd) pairs
    conc_x = np.concatenate(([0], x_values[:-1]))
    ad_arr = combos['ad'].to_numpy()
    bd_arr = combos['bd'].to_numpy()

    unique_pairs, inverse_idx = np.unique(
        np.stack([ad_arr, bd_arr], axis=1), axis=0, return_inverse=True)

    cdf_hi = beta.cdf(x_values[np.newaxis, :],
                      unique_pairs[:, 0:1], unique_pairs[:, 1:2])
    cdf_lo = beta.cdf(conc_x[np.newaxis, :],
                      unique_pairs[:, 0:1], unique_pairs[:, 1:2])
    dens_u = cdf_hi - cdf_lo
    dens_u /= dens_u.sum(axis=1, keepdims=True)

    m_matrix = dens_u[inverse_idx]   # (n_rows, intervals)

    # Calc betabinom dists – one matrix multiply per unique Count value
    counts_arr = combos['Count'].to_numpy(dtype=int)
    pmf_cache = {}
    for cc in np.unique(counts_arr):
        k = np.arange(cc + 1)[:, np.newaxis]
        pmf_cache[cc] = binom.pmf(k, n=cc, p=x_values)

    M_list = [pmf_cache[cc].dot(m_row) for cc, m_row in zip(counts_arr, m_matrix)]

    # Calc CDF of betabinoms and use to calc Vmin/max
    vmins = np.empty(len(combos), dtype=int)
    vmaxs = np.empty(len(combos), dtype=int)
    for i, M_arr in enumerate(M_list):
        cdf = np.cumsum(M_arr)
        cdf = np.where(cdf > 0.999999, 1.0, cdf)   # equivalent to round_numbers
        vmins[i] = int(np.argmax(cdf > alpha))
        vmaxs[i] = int(np.argmax(cdf > (1 - alpha)))

    combos['Vmin'] = vmins
    combos['Vmax'] = vmaxs

    # Correct error where Vmax is assigned 0 due to precision error
    combos.loc[combos['Vmax'] == 0, 'Vmax'] = combos.loc[combos['Vmax'] == 0, 'Count']

    # Assign each d node an E category – vectorised
    vmin = combos['Vmin'].to_numpy()
    vmax = combos['Vmax'].to_numpy()
    cnt  = combos['Count'].to_numpy()
    combos['E_Cat'] = np.select(
        [(vmin > 0) & (vmax < cnt),
         (vmin == 0) & (vmax < cnt),
         (vmin > 0) & (vmax == cnt),
         (vmin == 0) & (vmax == cnt)],
        ['Enf', 'E0', 'E1', 'Eam'],
        default='0'
    )

    print(combos['E_Cat'].value_counts())
    print(str(round(100*combos[combos['E_Cat'] == 'Enf']['Count'].sum()/combos['Count'].sum(), 1)) + '% of the data is in Enf')

    # Assign R, W and O for use later when determining feasibility of constraints – vectorised
    e_cat = combos['E_Cat'].to_numpy()
    combos['R'] = 0
    combos['W'] = combos['Count'].copy()
    combos['O'] = 1
    mask_e0_e1 = (e_cat == 'E0') | (e_cat == 'E1')
    mask_e1    = (e_cat == 'E1')
    combos.loc[mask_e0_e1, 'W'] = 0
    combos.loc[mask_e0_e1, 'O'] = 0
    combos.loc[mask_e1,    'R'] = combos.loc[mask_e1, 'Count']

    # Store m and M as object columns
    combos['m'] = 0
    combos['m'] = combos['m'].astype(object)
    for i, m_row in enumerate(m_matrix):
        combos.at[i, 'm'] = m_row

    combos['M'] = 0
    combos['M'] = combos['M'].astype(object)
    for i, M_arr in enumerate(M_list):
        combos.at[i, 'M'] = M_arr

    # Sort columns based on the number of categories ascending (makes code more efficient later)
    first_n_columns = combos.iloc[:, :(len(data.columns)-1)]
    remaining_columns = combos.iloc[:, (len(data.columns)-1):]
    unique_counts = first_n_columns.apply(lambda col: col.nunique())
    sorted_columns = unique_counts.sort_values().index
    combos = pd.concat([first_n_columns[sorted_columns], remaining_columns], axis=1)

    combos = combos.sort_values('Count', ascending=False).reset_index(drop=True)

    # Relax bounds for Enf nodes as they can't be predicted upon
    combosx = combos.copy()
    enf_mask = combos['E_Cat'] == 'Enf'
    combos.loc[enf_mask, 'Vmin'] = 0
    combos.loc[enf_mask, 'Vmax'] = combos.loc[enf_mask, 'Count']

    combos.to_pickle('combos.pkl')

    return combos, combosx

def round_numbers(arr):
    return [round(num, 1) if num > 0.999999 else num for num in arr]

def GenVs(dndf: pd.DataFrame, protected_vars: list[str]) -> pd.DataFrame:
    """
    Build the v‑node table for the decision‑node dataframe.

    Parameters
    ----------
    dndf : pd.DataFrame
        Original decision‑node table.
    protected_vars : list[str]
        Columns that must be *partially* specified in every v‑node.

    Returns
    -------
    pd.DataFrame
        v‑node table with per‑node aggregates and `direct_vchildren`.
        Ensures that 'm' is the last column.
    """
    print("\nGenVs")

    # ------------------------------------------------------------------ #
    # 1.  Pre‑processing                                                 #
    # ------------------------------------------------------------------ #
    dndf = dndf.drop(["E_Cat", "Vmin", "Vmax"], axis=1)

    cat_cols = dndf.columns[: dndf.columns.get_loc("Count")]
    num_cols = ["Count", "Target", "R", "W"]

    str_cat = dndf[cat_cols].astype(str).values
    n_d, n_cat = str_cat.shape

    # ------------------------------------------------------------------ #
    # 2.  Generate all incomplete node patterns (v‑nodes) – vectorised  #
    # ------------------------------------------------------------------ #
    keep_mask = np.array(list(itertools.product([False, True], repeat=n_cat)),
                         dtype=bool)  # (2^n_cat, n_cat)
    n_perms = keep_mask.shape[0]

    flat = np.empty((n_d * n_perms, n_cat), dtype=object)
    for c in range(n_cat):
        col_vals  = str_cat[:, c]
        col_block = np.where(keep_mask[np.newaxis, :, c],
                             col_vals[:, np.newaxis], -1)
        flat[:, c] = col_block.ravel()

    vns_df = pd.DataFrame(flat, columns=cat_cols).drop_duplicates()
    vns    = vns_df.values

    all_neg1 = np.array([(row == -1).all() for row in vns])
    vns = vns[~all_neg1]

    prot_idx = [list(cat_cols).index(p) for p in protected_vars]
    has_prot = np.array([(row[prot_idx] != -1).any() for row in vns])
    vns = vns[has_prot]
    n_v = len(vns)

    # ------------------------------------------------------------------ #
    # 3.  Aggregate source rows for every v‑node – bitmask grouping     #
    # ------------------------------------------------------------------ #
    num_arr  = dndf[num_cols].values.astype(int)
    sums_mat = np.zeros((n_v, len(num_cols)), dtype=int)
    dchildren = [None] * n_v

    specified_mat = (vns != -1).astype(bool)
    spec_keys = specified_mat.dot(1 << np.arange(n_cat))

    for bitmask in np.unique(spec_keys):
        rows      = np.where(spec_keys == bitmask)[0]
        spec_cols = np.where((bitmask >> np.arange(n_cat)) & 1)[0]

        if len(spec_cols) == 0:
            total   = num_arr.sum(axis=0)
            all_idx = list(range(n_d))
            for ri in rows:
                sums_mat[ri]  = total
                dchildren[ri] = all_idx
            continue

        d_vals = str_cat[:, spec_cols]
        v_vals = vns[rows][:, spec_cols]
        match  = (d_vals[:, np.newaxis, :] == v_vals[np.newaxis, :, :]).all(axis=2)
        sums_mat[rows] = match.T.astype(int) @ num_arr
        for j, ri in enumerate(rows):
            dchildren[ri] = np.nonzero(match[:, j])[0].tolist()

    df_cat = pd.DataFrame(vns, columns=cat_cols)
    df_num = pd.DataFrame(sums_mat, columns=num_cols, dtype=int)
    df_all = pd.concat([df_cat, df_num], axis=1)
    df_all["dchildren"] = dchildren

    vndf = (
        df_all[df_all["dchildren"].apply(len) > 1]
        .reset_index(drop=True)
        .assign(Vmin=0, Vmax=0, Fail=0, m=0)
    )

    # ------------------------------------------------------------------ #
    # 4.  Find direct v‑children – sort-based integer hash              #
    # ------------------------------------------------------------------ #
    vmat    = vndf[cat_cols].values
    levels  = (vmat == -1).sum(axis=1).astype(int)
    max_lvl = int(levels.max())
    buckets = [np.where(levels == lvl)[0] for lvl in range(max_lvl + 1)]

    vmat_int = np.full((len(vndf), n_cat), -1, dtype=np.int32)
    for c in range(n_cat):
        col      = vmat[:, c]
        str_mask = np.array([isinstance(v, str) for v in col])
        if str_mask.any():
            uniq_str   = np.unique(col[str_mask])
            val_to_int = {v: i for i, v in enumerate(uniq_str)}
            for ri in np.nonzero(str_mask)[0]:
                vmat_int[ri, c] = val_to_int[col[ri]]

    vmat_shifted = (vmat_int + 1).astype(np.int64)
    col_maxes    = vmat_shifted.max(axis=0) + 1
    strides      = np.ones(n_cat, dtype=np.int64)
    for c in range(1, n_cat):
        strides[c] = strides[c - 1] * int(col_maxes[c - 1])

    direct_list = [[] for _ in range(len(vndf))]

    for lvl in range(1, max_lvl + 1):
        parents  = buckets[lvl]
        children = buckets[lvl - 1]
        if len(children) == 0 or len(parents) == 0:
            continue

        parent_int     = vmat_int[parents]
        parent_shifted = vmat_shifted[parents]
        child_shifted  = vmat_shifted[children]

        parent_spec = (parent_int != -1)
        parent_bm   = parent_spec.dot(1 << np.arange(n_cat))

        for pbitmask in np.unique(parent_bm):
            p_local   = np.where(parent_bm == pbitmask)[0]
            p_global  = parents[p_local]
            sc        = np.where((pbitmask >> np.arange(n_cat)) & 1)[0]

            if len(sc) == 0:
                cl = children.tolist()
                for pi in p_global:
                    direct_list[pi] = cl
                continue

            sc_strides = strides[sc]
            c_hashes   = child_shifted[:, sc].dot(sc_strides)
            p_hashes   = parent_shifted[p_local][:, sc].dot(sc_strides)

            sort_idx        = np.argsort(c_hashes, kind='stable')
            sorted_hashes   = c_hashes[sort_idx]
            sorted_children = children[sort_idx]

            for li, gi in enumerate(p_global):
                h  = p_hashes[li]
                lo = int(np.searchsorted(sorted_hashes, h, side='left'))
                hi = int(np.searchsorted(sorted_hashes, h, side='right'))
                if lo < hi:
                    direct_list[gi] = sorted_children[lo:hi].tolist()

    vndf["direct_vchildren"] = direct_list

    # ------------------------------------------------------------------ #
    # 5.  Final tidy‑up                                                  #
    # ------------------------------------------------------------------ #
    mask_empty = vndf["direct_vchildren"].apply(len) == 0
    vndf.loc[mask_empty, "direct_vchildren"] = 0

    vndf["m"] = vndf["m"].astype(object)

    cols = [c for c in vndf.columns if c != "m"] + ["m"]
    vndf = vndf[cols]

    return vndf

def _get_pau(c1, c2, x_values, conc_x, cache):
    """
    Memoised beta‑CDF difference:
        pau(k) =  F_beta(x_values; c1,c2)  –  F_beta(conc_x; c1,c2)
    """
    key = (int(c1), int(c2))
    if key not in cache:
        density  = betainc(c1, c2, x_values) - betainc(c1, c2, conc_x)
        density /= density.sum()
        cache[key] = np.abs(density)
    return cache[key]

def _get_pmf(cc, x_values, cache):
    """
    Memoised binomial PMF table of shape  (cc+1, len(x_values)).
    """
    if cc not in cache:
        k = np.arange(cc + 1)[:, None]      # (k,1)
        cache[cc] = binom.pmf(k, n=cc, p=x_values)
    return cache[cc]

_LINSPACE_MS = np.linspace(0, 1, intervals)

def _batch_merge(rng, pta_mat, ptb_mat, c1_arr, c2_arr,
                 x_values, conc_x, density_cache, n_samples=5000):
    """
    Batch‑merge N pairs of distributions via Monte Carlo.
    Draws n_samples per distribution (>> K) then bins back to K bins,
    reducing histogram variance without changing the output shape.
    Uses a vectorised searchsorted (no Python loop over N rows).
    Returns (hist_mat (N, K), merged_counts (N,)).
    """
    N, K = pta_mat.shape
    S = n_samples
    pau_mat = np.empty((N, K), dtype=float)
    for n in range(N):
        pau_mat[n] = _get_pau(int(c1_arr[n]), int(c2_arr[n]),
                              x_values, conc_x, density_cache)

    row_off_K = (np.arange(N) * K)  # (N,) — for bincount offset

    def _sample(mat):
        cdf = np.cumsum(mat, axis=1)
        cdf[:, -1] = 1.0
        # Shift each row into a non-overlapping range [2n, 2n+1] so the
        # flattened array is globally sorted → single vectorised searchsorted
        row_shift = (np.arange(N) * 2.0)[:, None]   # (N, 1)
        cdf_flat = (cdf + row_shift).ravel()          # already globally sorted
        u_flat = (rng.random((N, S)) + row_shift).ravel()
        global_idx = np.searchsorted(cdf_flat, u_flat, side='right')
        local_idx = global_idx.reshape(N, S) - row_off_K[:, None]
        return _LINSPACE_MS[np.clip(local_idx, 0, K - 1)]

    x_mat = _sample(pta_mat)
    y_mat = _sample(ptb_mat)
    z_mat = _sample(pau_mat)

    w_mat = x_mat * z_mat + y_mat * (1.0 - z_mat)
    bin_idx = np.clip((w_mat * K).astype(np.int32), 0, K - 1)
    row_off = row_off_K[:, np.newaxis]
    hist_mat = (np.bincount((bin_idx + row_off).ravel(), minlength=N * K)
                .reshape(N, K).astype(float) / S)
    return hist_mat, c1_arr + c2_arr


def _resolve_dcs_batch(indices, dvc_arr, dch_arr, vndf_cats, cat_size_arr):
    """Return list of (src, rows) for each node in *indices*."""
    n_cat = vndf_cats.shape[1]
    result = []
    for i in indices:
        dvc = dvc_arr[i]
        if not (isinstance(dvc, (int, float)) and dvc == 0):
            cand_rows = dvc
            cand_cats = vndf_cats[cand_rows]
            found = False
            for c_idx in range(n_cat):
                col_vals = cand_cats[:, c_idx]
                uniq = np.unique(col_vals[col_vals != -1])
                if uniq.size == cat_size_arr[c_idx]:
                    keep = col_vals != -1
                    result.append(('vndf',
                                   np.array(cand_rows)[keep].tolist()))
                    found = True
                    break
            if not found:
                dch = dch_arr[i]
                result.append(('dndf',
                               dch if hasattr(dch, '__len__') else [dch]))
        else:
            dch = dch_arr[i]
            result.append(('dndf',
                           dch if hasattr(dch, '__len__') else [dch]))
    return result


def _splitting_solver_seq(m_list, c_list, rng, x_values, conc_x, cache,
                          n_samples=5000):
    """Binary‑reduction tree on raw arrays (no DataFrame overhead)."""
    K = len(m_list[0])
    S = n_samples
    while len(m_list) > 1:
        new_m, new_c = [], []
        for j in range(0, len(m_list), 2):
            if j + 1 < len(m_list):
                pta, ptb = m_list[j], m_list[j + 1]
                c1, c2 = c_list[j], c_list[j + 1]
                pau = _get_pau(c1, c2, x_values, conc_x, cache)
                cdf_a = np.cumsum(pta); cdf_a[-1] = 1.0
                cdf_b = np.cumsum(ptb); cdf_b[-1] = 1.0
                cdf_u = np.cumsum(pau); cdf_u[-1] = 1.0
                u = rng.random(S)
                x = _LINSPACE_MS[np.clip(
                    np.searchsorted(cdf_a, u, side='right'), 0, K - 1)]
                u = rng.random(S)
                y = _LINSPACE_MS[np.clip(
                    np.searchsorted(cdf_b, u, side='right'), 0, K - 1)]
                u = rng.random(S)
                z = _LINSPACE_MS[np.clip(
                    np.searchsorted(cdf_u, u, side='right'), 0, K - 1)]
                w = x * z + y * (1.0 - z)
                hist, _ = np.histogram(w, bins=K, range=[0, 1])
                new_m.append(hist / S)
                new_c.append(c1 + c2)
            else:
                new_m.append(m_list[j])
                new_c.append(c_list[j])
        m_list, c_list = new_m, new_c
    return m_list[0], c_list[0]


def CalcMs(data, dndf, vndf, hbaseline, x_values, seed=0, n_samples=5000):
    """
    Depends on global constants:
        intervals  –  histogram / discretisation length
        a0, b0     –  prior hyper‑parameters
    """
    print("CalcMs")

    # ---------------------------------------------------------------
    # One‑time arrays & caches
    # ---------------------------------------------------------------
    conc_x        = np.concatenate(([0], x_values[:-1]))
    density_cache = {}

    dndf   = dndf.drop(['E_Cat', 'Vmin', 'Vmax'], axis=1)
    cols   = dndf.columns[: dndf.columns.get_loc("Count")]
    lencols = len(cols)

    catvals   = {c: set(data[c].unique()) for c in cols}
    cat_size  = {c: len(v - {-1}) for c, v in catvals.items()}
    cat_size_arr = np.array([cat_size[c] for c in cols])

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    # Pre‑extract arrays (avoid repeated iloc)
    # ---------------------------------------------------------------
    dvc_arr        = vndf['direct_vchildren'].values
    dch_arr        = vndf['dchildren'].values
    count_vndf     = vndf['Count'].values
    vndf_cats      = vndf[cols].values
    dndf_m_arr     = np.stack(dndf['m'].values)
    dndf_count_arr = dndf['Count'].values
    m_vndf         = vndf['m'].values.copy()

    def _get_m_count(src, row_idx):
        if src == 'vndf':
            return (np.asarray(m_vndf[row_idx], dtype=float),
                    int(count_vndf[row_idx]))
        return dndf_m_arr[row_idx].copy(), int(dndf_count_arr[row_idx])

    # ---------------------------------------------------------------
    # 1.  Process nodes level by level
    # ---------------------------------------------------------------
    for l in range(1, lencols):
        print(l)
        indices = np.where((vndf_cats == -1).sum(axis=1) == l)[0]

        dcs_info = _resolve_dcs_batch(
            indices, dvc_arr, dch_arr, vndf_cats, cat_size_arr)

        # Group by dcs size
        by_size = {}
        for pos, (src, rows) in enumerate(dcs_info):
            by_size.setdefault(len(rows), []).append(
                (indices[pos], src, rows))

        for n_dcs in sorted(by_size.keys()):
            nodes = by_size[n_dcs]
            N = len(nodes)
            if n_dcs < 2:
                continue

            # Extract per‑row m and count arrays
            m_rows, c_rows = [], []
            for j in range(n_dcs):
                mj, cj = [], []
                for (vi, src, rows) in nodes:
                    m, c = _get_m_count(src, rows[j])
                    mj.append(m); cj.append(c)
                m_rows.append(np.stack(mj))
                c_rows.append(np.array(cj))

            if n_dcs <= 4:
                # Batched binary reduction tree
                while len(m_rows) > 1:
                    new_m, new_c = [], []
                    for j in range(0, len(m_rows), 2):
                        if j + 1 < len(m_rows):
                            merged, counts = _batch_merge(
                                rng, m_rows[j], m_rows[j + 1],
                                c_rows[j], c_rows[j + 1],
                                x_values, conc_x, density_cache,
                                n_samples=n_samples)
                            new_m.append(merged)
                            new_c.append(counts)
                        else:
                            new_m.append(m_rows[j])
                            new_c.append(c_rows[j])
                    m_rows, c_rows = new_m, new_c
                final_mat = m_rows[0]
                for n, (vi, _, _) in enumerate(nodes):
                    m_vndf[vi] = final_mat[n]
            else:
                # Size ≥ 5: sequential reduction (rare)
                for n, (vi, src, rows) in enumerate(nodes):
                    ml = [m_rows[j][n] for j in range(n_dcs)]
                    cl = [c_rows[j][n] for j in range(n_dcs)]
                    m_val, _ = _splitting_solver_seq(
                        ml, cl, rng, x_values, conc_x, density_cache,
                        n_samples=n_samples)
                    m_vndf[vi] = m_val

    # Write m back to vndf
    for i in range(len(vndf)):
        vndf.at[i, 'm'] = m_vndf[i]

    # ---------------------------------------------------------------
    # 2.  Betabinomial approximation – vectorised by cc group
    # ---------------------------------------------------------------
    vndf['M'] = None
    pmf_cache = {}
    counts_arr = vndf['Count'].to_numpy(dtype=int)
    cc_arr     = np.where(counts_arr > hbaseline, hbaseline, counts_arr)

    for cc in np.unique(cc_arr):
        pmf_vals = _get_pmf(int(cc), x_values, pmf_cache)
        rows     = np.where(cc_arr == cc)[0]
        m_stack  = np.stack(
            [np.asarray(m_vndf[int(r)], dtype=float) for r in rows])
        M_mat    = pmf_vals.dot(m_stack.T).T

        for j, r in enumerate(rows):
            count_val = int(counts_arr[r])
            M_arr = M_mat[j]
            if count_val > hbaseline:
                iv = np.interp(np.linspace(0, 1, count_val),
                               np.linspace(0, 1, cc + 1), M_arr)
                iv /= iv.sum()
                vndf.at[int(r), 'M'] = iv.tolist()
            else:
                vndf.at[int(r), 'M'] = M_arr.tolist()

    # ---------------------------------------------------------------
    # 3.  Adjustment variable 't'
    # ---------------------------------------------------------------
    vndf['t'] = (
        vndf['Count'] * (vndf['Target'] + a0) / (vndf['Count'] + a0 + b0)
    ).round()

    return vndf

def CalcBounds(vndf, alpha):
    print('CalcBounds')

    # 1) Compute all CDFs in one go
    Ms = vndf['M'].tolist()
    cdf_list = [np.cumsum(m) for m in Ms]
    vndf['cdf_M'] = cdf_list

    # 2) Vectorized Vmin/Vmax via argmax on each CDF
    vmins = np.array([int(np.argmax(c > alpha))     for c in cdf_list])
    vmaxs = np.array([int(np.argmax(c > (1-alpha))) for c in cdf_list])

    # Fix any zero‐due‐to‐precision cases
    zero_mask = (vmaxs == 0)
    if zero_mask.any():
        vmaxs[zero_mask] = vndf.loc[zero_mask, 'Count'].values

    vndf['Vmin'] = vmins
    vndf['Vmax'] = vmaxs

    # 3) Compute adjustval as FLOAT (still rounded to nearest integer)
    adjust = np.round(vndf['t'] - (vmins + vmaxs) / 2, 0)   # this yields floats like 1.0, -2.0, etc.
    vndf['adjustval'] = adjust

    # 4) Apply boundary adjustments only where needed
    mask = (vndf['Vmin'] != 0) & (vndf['Vmax'] != vndf['Count'])
    if mask.any():
        vndf.loc[mask, 'Vmin'] = np.maximum(vndf.loc[mask, 'Vmin'] + adjust[mask], 0)
        vndf.loc[mask, 'Vmax'] = np.minimum(vndf.loc[mask, 'Vmax'] + adjust[mask],
                                            vndf.loc[mask, 'Count'])

    # 5) Compute Fail vectorized
    feasible = (vndf['Vmin'] <= (vndf['R'] + vndf['W'])) & (vndf['Vmax'] >= vndf['R'])
    vndf['Fail'] = (~feasible).astype(int)

    return vndf

def AdjustEnfs(vndf, dndf):
    print('AdjustEnfs')
    # initial row count
    initial_n = len(vndf)
    print(initial_n)

    # 1) drop unused columns
    vndf = vndf.drop(['m', 'adjustval', 't'], axis=1)

    # 2) filter to only those nodes needing adjustment
    mask_keep = (vndf['Vmin'] != 0) | (vndf['Vmax'] != vndf['Count'])
    vndf = vndf.loc[mask_keep].reset_index(drop=True)
    filtered_n = len(vndf)
    print(filtered_n)

    # 3) precompute, for each d‑node, how many 'Enf' counts it has
    #    this is a 1D array aligned with dndf.index
    enf_counts = dndf['Count'].where(dndf['E_Cat'] == 'Enf', 0).to_numpy()

    # 4) vectorized sum of enf_counts over each v‑node’s dchildren list
    dchildren_lists = vndf['dchildren'].tolist()
    sum_enf = np.array([enf_counts[inds].sum() for inds in dchildren_lists])

    # 5) build mask of which v‑nodes exceed 50% Enf domination
    counts = vndf['Count'].to_numpy()
    mask_dom = sum_enf > 0.5 * counts

    if mask_dom.any():
        # grab Vmin/Vmax as numpy arrays for fast arithmetic
        vmin = vndf['Vmin'].to_numpy()
        vmax = vndf['Vmax'].to_numpy()

        # compute adjusted boundaries
        new_vmin = np.maximum(vmin[mask_dom] - sum_enf[mask_dom], 0)
        new_vmax = np.minimum(vmax[mask_dom] + sum_enf[mask_dom], counts[mask_dom])

        # write back in one go
        vndf.loc[mask_dom, 'Vmin'] = new_vmin
        vndf.loc[mask_dom, 'Vmax'] = new_vmax

    # 6) final row count (should equal filtered_n)
    print(len(vndf))
    return vndf

def compute_proportions(df, protected_vars,
                        target_col='E_Cat',
                        target_value='Enf',
                        count_col='Count'):
    """
    Prints, for every combination of protected attributes,
    1) total sample count,
    2) count of rows where target_col == target_value,
    3) proportion of those rows within the group.
    """
    # total rows per subgroup
    total_counts = (
        df.groupby(protected_vars)[count_col]
          .sum()
          .rename('Total')
    )

    # rows in the target category per subgroup
    enf_counts = (
        df[df[target_col] == target_value]
          .groupby(protected_vars)[count_col]
          .sum()
          .reindex(total_counts.index, fill_value=0)
          .rename('Enf')
    )

    # proportion (with division-by-zero protection)
    proportions = (enf_counts / total_counts).fillna(0).rename('Prop')

    # combine into one frame for clarity
    summary = pd.concat([total_counts, enf_counts, proportions], axis=1)

    # pretty print
    for (attr1, attr2), row in summary.iterrows():
        print(
            f"1st {protected_vars[0]} {attr1} "
            f"2nd {protected_vars[1]} {attr2} "
            f"in {target_value}: "
            f"count {int(row['Enf'])} / {int(row['Total'])} "
            f"({row['Prop']:.2%})"
        )

    return summary  # optional: return the DataFrame for downstream use

def ttpreprocess_data(df, target_col='target', columns_reference=None):
    """
    Preprocesses the DataFrame by one-hot encoding categorical columns (excluding the target)
    and converting features and target into appropriate data types.
    
    If columns_reference is provided, reindex the one-hot encoded DataFrame to match it,
    ensuring that both training and test sets have the same columns.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.drop(target_col, errors='ignore')
    ohe_df = pd.get_dummies(df, columns=categorical_cols)
    
    # If a reference set of columns is provided (from training data), align the columns of the current DataFrame.
    if columns_reference is not None:
        ohe_df = ohe_df.reindex(columns=columns_reference, fill_value=0)
    
    X = ohe_df.drop(target_col, axis=1).values.astype('float32')
    y = ohe_df[target_col].values.astype('int')
    return X, y, ohe_df.columns

def ttassign_prediction(row, ref_df, feature_cols):
    """
    Given a row from the new test DataFrame, check if the combination of feature values
    (from feature_cols) appears in the reference DataFrame (ref_df). 
    
    If it appears:
        - If any matching row's E_Cat is 'Enf', return 'NF'.
        - Otherwise, if the matching row's N_Pred0 is 0, return 0; if non-zero, return 1.
    
    If it does not appear, return 'New'.
    """
    # Create a boolean mask for rows in ref_df that match the new row's feature values
    mask = pd.Series([True] * len(ref_df))
    for col in feature_cols:
        mask &= (ref_df[col] == row[col])
        
    matching_rows = ref_df[mask]
    
    if matching_rows.empty:
        return 'New'
    else:
        # If any match has E_Cat equal to 'Enf', assign 'NF'
        if (matching_rows['E_Cat'] == 'Enf').any():
            return 'NF'
        else:
            # Otherwise, check the value of N_Pred0 in the first matching row
            n_pred0 = matching_rows.iloc[0]['N_Pred0']
            return 0 if n_pred0 == 0 else 1



