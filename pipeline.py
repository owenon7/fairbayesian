"""
FairBayesian JAIR pipeline: reusable library.

Extracted from FairBayesian4.ipynb cells 4-6 so the notebook and the CLI
driver (run_experiments.py) call the same code. Do not edit the mathematical
body of build_and_solve / score_vnode_ll without running verify_uniqueness.py
afterwards -- the JAIR paper's uniqueness story depends on them.

Public API:

    compute_leaf_weights(dndf)                -> np.ndarray
    build_and_solve(vndf, dndf, ...)          -> (model, x, n_constr)
    build_vll_scorer(vndf_base, dndf)         -> (score_fn, valid_idx, A_csr)
    solve_and_select(vndf, vndf_base, dndf,
                     pool_solutions=100, ...) -> SolveResult
    run_baselines(data, protected_vars, ...)  -> (res_df, meta)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# Prefer the user's academic licence at ~/gurobi.lic over the size-limited
# licence that ships bundled with gurobipy (`pip install gurobipy` sets
# GRB_LICENSE_FILE to that in some shells; it would otherwise cap us at 2000
# variables, which is far too small for Adult / Bank Marketing).
_USER_LIC = os.path.expanduser("~/gurobi.lic")
if os.path.exists(_USER_LIC):
    os.environ["GRB_LICENSE_FILE"] = _USER_LIC

import gurobipy as gp  # noqa: E402
from gurobipy import GRB  # noqa: E402
from scipy.sparse import lil_matrix  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# D-node log-posterior weights (MIP objective)
# ──────────────────────────────────────────────────────────────────────────

def compute_leaf_weights(dndf: pd.DataFrame) -> np.ndarray:
    """Per d-node weight ``w_i = log(p_i / (1 - p_i)) * log(Count_i + 1)``.

    ``p_i = E[K] / Count_i`` is the posterior-mean positive rate computed
    from the beta-binomial distribution ``M_i``. Used as the MIP objective
    weight in build_and_solve. Returned weights are 0 for zero-count nodes.
    """
    n = len(dndf)
    w = np.zeros(n)
    counts = dndf["Count"].to_numpy()
    M_col = dndf["M"].to_numpy()
    for i in range(n):
        count = int(counts[i])
        if count == 0:
            continue
        M_arr = np.asarray(M_col[i], dtype=float)
        mean_k = float(np.dot(np.arange(len(M_arr)), M_arr))
        p = np.clip(mean_k / count, 1e-10, 1 - 1e-10)
        w[i] = np.log(p / (1 - p)) * np.log(count + 1)
    return w


# ──────────────────────────────────────────────────────────────────────────
# MIP build + solve
# ──────────────────────────────────────────────────────────────────────────

def build_and_solve(
    vndf: pd.DataFrame,
    dndf: pd.DataFrame,
    pool_solutions: int = 100,
    seed: int = 42,
    time_limit: float | None = None,
    verbose: bool = False,
) -> tuple[gp.Model, dict[int, gp.Var], int]:
    """Build and solve the FairBayesian MIP with a D-node log-posterior objective.

    Returns ``(model, x, n_constraints)``. ``x[i]`` is the binary prediction
    variable for d-node ``i``. Pool-search-mode 2 is set so the returned pool
    contains the top `pool_solutions` solutions by objective.
    """
    indices = list(range(len(dndf)))
    active_set = set(dndf[dndf["O"] != 0].index)
    active_idx = list(active_set)

    model = gp.Model("FairBayes")
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.setParam("Seed", seed)
    model.setParam("PoolSearchMode", 2)
    model.setParam("PoolSolutions", pool_solutions)
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))

    x = model.addVars(indices, vtype=GRB.BINARY, name="x")

    # Fix d_0 and d_1 nodes
    for i in indices:
        if dndf["O"].iloc[i] == 0:
            val = 1 if dndf["R"].iloc[i] > 0 else 0
            x[i].lb = x[i].ub = val

    # V-node constraints: Vmin(v) ≤ Σ_{i ∈ dchildren(v)} N_{d_i} · x_i ≤ Vmax(v)
    # where x_i ∈ {0,1} is the d-node decision and N_{d_i} = Count_{d_i}.
    # The LHS equals Σ P_{d_i} under determinism (P_d = x_d · N_d).
    n_constr = 0
    dc_arr = vndf["dchildren"].values
    vmin_arr = vndf["Vmin"].values
    vmax_arr = vndf["Vmax"].values
    count_col = dndf["Count"].to_numpy(dtype=int)
    for i in range(len(vndf)):
        dc = dc_arr[i]
        if not any(c in active_set for c in dc):
            continue
        svars: list[gp.Var] = []
        scoeffs: list[int] = []
        for idx in dc:
            n = int(count_col[idx])
            if n != 0:
                svars.append(x[idx])
                scoeffs.append(n)
        if not svars:
            continue
        lhs = gp.LinExpr(scoeffs, svars)
        if vmin_arr[i] > 0:
            model.addConstr(lhs >= int(vmin_arr[i]))
            n_constr += 1
        model.addConstr(lhs <= int(vmax_arr[i]))
        n_constr += 1

    # D-node (leaf) bounds
    dn_vmin = dndf["Vmin"].to_numpy(dtype=int)
    dn_vmax = dndf["Vmax"].to_numpy(dtype=int)
    for i in active_idx:
        if dn_vmin[i] > 0:
            model.addConstr(x[i] >= int(dn_vmin[i]))
            n_constr += 1
        model.addConstr(x[i] <= int(dn_vmax[i]))
        n_constr += 1

    # Objective: Σ w_i · x_i (maximise)
    w = compute_leaf_weights(dndf)
    model.setObjective(
        gp.quicksum(w[i] * x[i] for i in active_idx),
        GRB.MAXIMIZE,
    )

    model.optimize()
    return model, x, n_constr


# ──────────────────────────────────────────────────────────────────────────
# V-node log-likelihood scoring
# ──────────────────────────────────────────────────────────────────────────

EPSILON = 1e-300  # floor to avoid log(0) in M_v lookups


def build_vll_scorer(vndf_base: pd.DataFrame, dndf: pd.DataFrame):
    """Precompute the adjacency matrix and per-v-node M arrays.

    Returns ``(score_fn, valid_idx, A_csr)`` where ``score_fn(sol_vec)``
    evaluates ``Σ_v log M_v[t_v]`` over v-nodes with valid M distributions.
    Only v-nodes whose M is an array (not scalar/None) are included.
    """
    M_arr_base = vndf_base["M"].values
    dc_base = vndf_base["dchildren"].values
    target_d = dndf["Target"].values.astype(int)
    n_d = len(dndf)

    valid_mask = np.array([
        m is not None and not isinstance(m, (int, float)) and hasattr(m, "__len__")
        for m in M_arr_base
    ])
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)

    A = lil_matrix((n_valid, n_d), dtype=int)
    for row, vi in enumerate(valid_idx):
        for di in dc_base[vi]:
            A[row, di] = target_d[di]
    A_csr = A.tocsr()

    valid_Ms = [np.asarray(M_arr_base[vi], dtype=float) for vi in valid_idx]
    valid_Mlens = np.array([len(m) for m in valid_Ms])
    # Pre-log M arrays so the hot loop does a single array lookup per v-node.
    valid_logMs = [np.log(np.maximum(m, EPSILON)) for m in valid_Ms]

    def score_fn(sol_vec: np.ndarray) -> float:
        t_vec = A_csr @ sol_vec  # (n_valid,) implied target counts
        ll = 0.0
        for i in range(n_valid):
            t_v = int(t_vec[i])
            if t_v < 0:
                t_v = 0
            elif t_v >= valid_Mlens[i]:
                t_v = int(valid_Mlens[i] - 1)
            ll += valid_logMs[i][t_v]
        return float(ll)

    return score_fn, valid_idx, A_csr


# ──────────────────────────────────────────────────────────────────────────
# High-level convenience: solve, extract pool, rescore by V-LL
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SolveResult:
    """Outcome of a solve_and_select call."""

    feasible: bool
    n_constraints: int
    pool_size: int
    mip_gap: float | None
    runtime: float
    # Per-pool arrays (length pool_size). Absent/empty if infeasible.
    sols: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    d_obj: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v_ll: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Selector outcomes
    best_d_idx: int | None = None
    best_v_idx: int | None = None
    pred_d: np.ndarray | None = None
    pred_v: np.ndarray | None = None
    n_at_best_d: int = 0
    n_at_best_v: int = 0
    spearman_rho: float | None = None
    spearman_p: float | None = None
    # Bookkeeping
    extras: dict[str, Any] = field(default_factory=dict)


def solve_and_select(
    vndf: pd.DataFrame,
    vndf_base: pd.DataFrame,
    dndf: pd.DataFrame,
    pool_solutions: int = 100,
    seed: int = 42,
    time_limit: float | None = None,
    verbose: bool = False,
    score_fn=None,
) -> SolveResult:
    """Build the MIP, solve, extract the full pool, and rescore by V-node LL.

    If ``score_fn`` is provided (from build_vll_scorer), it is reused so we
    don't rebuild the adjacency matrix per call. Otherwise it is built once
    from ``vndf_base`` and ``dndf``.
    """
    t0 = time.time()
    model, x, n_constr = build_and_solve(
        vndf,
        dndf,
        pool_solutions=pool_solutions,
        seed=seed,
        time_limit=time_limit,
        verbose=verbose,
    )
    runtime = time.time() - t0

    mip_gap = float(model.MIPGap) if model.SolCount > 0 else None

    if model.SolCount == 0:
        return SolveResult(
            feasible=False,
            n_constraints=n_constr,
            pool_size=0,
            mip_gap=mip_gap,
            runtime=runtime,
        )

    if score_fn is None:
        score_fn, _, _ = build_vll_scorer(vndf_base, dndf)

    n_pool = model.SolCount
    n_d = len(dndf)
    sols = np.zeros((n_pool, n_d), dtype=int)
    d_obj = np.zeros(n_pool)
    for s in range(n_pool):
        model.setParam(GRB.Param.SolutionNumber, s)
        sols[s] = np.array([x[i].Xn for i in range(n_d)]).astype(int)
        d_obj[s] = model.PoolObjVal
    model.setParam(GRB.Param.SolutionNumber, 0)

    v_ll = np.array([score_fn(sols[s]) for s in range(n_pool)])

    best_d_idx = int(np.argmax(d_obj))
    best_v_idx = int(np.argmax(v_ll))

    # Uniqueness counts. Use absolute tolerance only (np.isclose defaults
    # include rtol=1e-5, which on a D-obj of ~500 admits ~5e-3 differences).
    # Matches the original FairBayesian4.ipynb Cell 6 tolerance of < 1e-6.
    n_at_best_d = int(np.sum(np.abs(d_obj - d_obj[best_d_idx]) < 1e-6))
    n_at_best_v = int(np.sum(np.abs(v_ll - v_ll[best_v_idx]) < 1e-6))

    rho = rho_p = None
    if n_pool > 2:
        rho_stat, rho_p_val = spearmanr(d_obj, v_ll)
        rho, rho_p = float(rho_stat), float(rho_p_val)

    return SolveResult(
        feasible=True,
        n_constraints=n_constr,
        pool_size=n_pool,
        mip_gap=mip_gap,
        runtime=runtime,
        sols=sols,
        d_obj=d_obj,
        v_ll=v_ll,
        best_d_idx=best_d_idx,
        best_v_idx=best_v_idx,
        pred_d=sols[best_d_idx].copy(),
        pred_v=sols[best_v_idx].copy(),
        n_at_best_d=n_at_best_d,
        n_at_best_v=n_at_best_v,
        spearman_rho=rho,
        spearman_p=rho_p,
    )


# ──────────────────────────────────────────────────────────────────────────
# Baselines: DT + NN + PMC
# ──────────────────────────────────────────────────────────────────────────

def _aggregate_to_dnodes(data: pd.DataFrame, pred: np.ndarray, pred_col: str) -> pd.DataFrame:
    """Aggregate per-instance predictions back to d-node level.

    Returns a DataFrame with the full feature tuple plus ``Count`` and
    ``<pred_col>`` (= number of positive predictions in that d-node).
    """
    df = data.drop(columns=["target"]).copy()
    df[pred_col] = pred

    feature_cols = [c for c in df.columns if c != pred_col]
    counts = (
        df[feature_cols]
        .value_counts()
        .rename("Count")
        .reset_index()
    )
    positives = (
        df.loc[df[pred_col] == 1, feature_cols]
        .value_counts()
        .rename(pred_col)
        .reset_index()
    )
    merged = counts.merge(positives, on=feature_cols, how="left").fillna(0)
    merged[pred_col] = merged[pred_col].astype(int)
    return merged


def run_baselines(
    data: pd.DataFrame,
    protected_vars: list[str],
    seed: int = 42,
    dt_grid: dict | None = None,
    nn_grid: dict | None = None,
    skip_pmc: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train DT, NN, and PMC baselines; return per-d-node predictions + CV meta.

    Default grids mirror FairBayesian4.ipynb cells 8-10.
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

    if dt_grid is None:
        dt_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": [None, "sqrt", "log2"],
            "ccp_alpha": [0.0, 0.001, 0.01, 0.1],
        }
    if nn_grid is None:
        nn_grid = {
            "hidden_layer_sizes": [(10, 10), (100, 10), (50, 50)],
            "activation": ["relu", "logistic"],
            "alpha": [1e-4, 1e-3],
            "solver": ["adam"],
            "learning_rate": ["adaptive"],
        }

    categorical_cols = (
        data.drop(columns=["target"]).select_dtypes(include="object").columns.tolist()
    )
    ohe_df = pd.get_dummies(data, columns=categorical_cols)
    X = ohe_df.drop(columns=["target"]).values.astype(int)
    y = ohe_df["target"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    cv_strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    meta: dict[str, Any] = {}

    # ── Decision Tree ──────────────────────────────────────────────────
    if verbose:
        print("[baselines] DT: GridSearchCV...")
    dt_search = GridSearchCV(
        DecisionTreeClassifier(random_state=seed),
        dt_grid, cv=cv_strat, scoring="accuracy", n_jobs=-1,
    )
    dt_search.fit(X_train, y_train)
    best_dt = DecisionTreeClassifier(**dt_search.best_params_, random_state=seed)
    best_dt.fit(X_train, y_train)
    y_pred_dt = best_dt.predict(X)
    meta["dt"] = {
        "best_params": dt_search.best_params_,
        "cv_accuracy": float(dt_search.best_score_),
        "test_accuracy": float(dt_search.score(X_test, y_test)),
        "train_positive_rate": float(y_pred_dt.mean()),
    }

    # ── Neural Net ─────────────────────────────────────────────────────
    if verbose:
        print("[baselines] NN: GridSearchCV...")
    nn_search = GridSearchCV(
        MLPClassifier(max_iter=200, random_state=seed),
        nn_grid, cv=cv_strat, scoring="accuracy", n_jobs=-1,
    )
    nn_search.fit(X_train, y_train)
    best_nn = MLPClassifier(**nn_search.best_params_, max_iter=200, random_state=seed)
    best_nn.fit(X_train, y_train)
    y_pred_nn = best_nn.predict(X)
    meta["nn"] = {
        "best_params": nn_search.best_params_,
        "cv_accuracy": float(nn_search.best_score_),
        "test_accuracy": float(nn_search.score(X_test, y_test)),
        "train_positive_rate": float(y_pred_nn.mean()),
    }

    # ── PMC (logistic regression + MultiCalibrator) ────────────────────
    if skip_pmc:
        y_pred_mc = np.zeros(len(y), dtype=int)
        meta["pmc"] = {"skipped": True}
    else:
        if verbose:
            print("[baselines] PMC: fitting...")
        from pmc import MultiCalibrator, Auditor

        ohe = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
            remainder="drop",
        )
        base_lr = Pipeline([
            ("ohe", ohe),
            ("lr", LogisticRegression(max_iter=500, solver="lbfgs")),
        ])
        mc_clf = MultiCalibrator(
            estimator=base_lr,
            auditor_type=Auditor(groups=protected_vars),
            n_bins=5,
            random_state=seed,
        )
        X_df = data.drop(columns=["target"])
        mc_clf.fit(X_df, y)
        p_mc = mc_clf.predict_proba(X_df)[:, 1]
        threshold = float(y.mean())
        y_pred_mc = (p_mc >= threshold).astype(int)
        meta["pmc"] = {
            "threshold": threshold,
            "train_positive_rate": float(y_pred_mc.mean()),
        }

    # ── Aggregate each to d-node level and merge ───────────────────────
    dt_res = _aggregate_to_dnodes(data, y_pred_dt, "DT_Pred")
    nn_res = _aggregate_to_dnodes(data, y_pred_nn, "NN_Pred")
    mc_res = _aggregate_to_dnodes(data, y_pred_mc, "MC_Pred")

    feature_cols = list(dt_res.columns.drop(["Count", "DT_Pred"]))
    merged = (
        dt_res.merge(nn_res.drop(columns=["Count"]), on=feature_cols, how="outer")
              .merge(mc_res.drop(columns=["Count"]), on=feature_cols, how="outer")
              .fillna(0)
    )
    for col in ("DT_Pred", "NN_Pred", "MC_Pred"):
        merged[col] = merged[col].astype(int)

    return merged, meta
