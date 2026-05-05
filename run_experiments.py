"""
JAIR results pipeline: per-dataset driver.

Runs the full FairBayesian pipeline (CalcMs, alpha sweep, alpha* solve with
both D-obj and V-LL selectors), trains the DT/NN/PMC baselines, and writes
every table, node-level dataframe, illustrative-example candidate list, and
figure the paper needs to results/<ds>/.

See jair-results-script-5ae072.md for the plan. See verify_uniqueness.py for
the pre-flight uniqueness checks this file assumes have passed.

Usage:
    python run_experiments.py --dataset all
    python run_experiments.py --dataset adult
    python run_experiments.py --dataset compas --no-alpha-sweep
    python run_experiments.py --dataset bm --skip-baselines
    python run_experiments.py --dataset all --force
    python run_experiments.py --dataset adult --skip-verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure we can import fb_functions & pipeline regardless of cwd
PHD_DIR = Path(__file__).resolve().parent
if str(PHD_DIR) not in sys.path:
    sys.path.insert(0, str(PHD_DIR))

from fb_functions import DataPrep, GenVs, CalcMs, CalcBounds, AdjustEnfs  # noqa: E402
from pipeline import (  # noqa: E402
    build_vll_scorer,
    solve_and_select,
    run_baselines,
)

from verify_uniqueness import DATASET_CONFIG, INTERVALS, HBASELINE, A0, B0, X_VALUES  # noqa: E402

ALPHA_SWEEP = [1e-3, 1e-4, 1e-5, 5e-6, 2e-6, 1.5e-6, 1e-6, 5e-7, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12]

POOL_SOLUTIONS = 100
BASELINE_SEED = 42
MIP_SEED = 42


# ──────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stage(label: str):
    """Context manager-ish stage printer (minimalist)."""
    class _Stage:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n[stage] {label} ...")
            return self

        def __exit__(self, *exc):
            dt = time.time() - self.t0
            print(f"[stage] {label} done ({dt:.1f}s)")
    return _Stage()


def _to_json_safe(obj):
    """Recursively convert numpy/pandas types to JSON-serialisable."""
    import numpy as _np
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.bool_):
        return bool(obj)
    return obj


# ──────────────────────────────────────────────────────────────────────────
# Cached CalcMs
# ──────────────────────────────────────────────────────────────────────────

def prepare_dataset(ds: str, results_root: Path, force: bool = False):
    cfg = DATASET_CONFIG[ds]
    data_path = PHD_DIR / cfg["parquet"]
    ds_dir = _ensure_dir(results_root / ds)
    cache_dir = _ensure_dir(ds_dir / "cache")

    data_cache = cache_dir / "data.parquet"
    dndf_cache = cache_dir / "dndf_base.pkl"
    vndf_cache = cache_dir / "vndf_base.pkl"

    if not force and dndf_cache.exists() and vndf_cache.exists() and data_cache.exists():
        with _stage(f"[{ds}] load cached DataPrep/GenVs/CalcMs"):
            data = pd.read_parquet(data_cache)
            dndf = pd.read_pickle(dndf_cache)
            vndf_base = pd.read_pickle(vndf_cache)
    else:
        with _stage(f"[{ds}] DataPrep + GenVs + CalcMs (slow)"):
            data = pd.read_parquet(data_path)
            dndf, _ = DataPrep(data, A0, B0, X_VALUES)
            vndf = GenVs(dndf, cfg["protected_vars"])
            vndf = CalcMs(data, dndf, vndf, HBASELINE, X_VALUES)
            vndf_base = vndf.copy()
            data.to_parquet(data_cache)
            dndf.to_pickle(dndf_cache)
            vndf_base.to_pickle(vndf_cache)
    return data, dndf, vndf_base, cfg, ds_dir


def cache_or_run_baselines(
    data: pd.DataFrame,
    protected_vars: list[str],
    ds_dir: Path,
    force: bool = False,
    skip_pmc: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cache = ds_dir / "cache" / "baselines.pkl"
    if not force and cache.exists():
        with _stage("[baselines] load cached"):
            obj = pd.read_pickle(cache)
        return obj["res"], obj["meta"]
    with _stage("[baselines] train DT + NN + PMC"):
        res, meta = run_baselines(
            data,
            protected_vars,
            seed=BASELINE_SEED,
            skip_pmc=skip_pmc,
        )
        pd.to_pickle({"res": res, "meta": meta}, cache)
    return res, meta


# ──────────────────────────────────────────────────────────────────────────
# Alpha sweep
# ──────────────────────────────────────────────────────────────────────────

def alpha_sweep(
    dndf: pd.DataFrame,
    vndf_base: pd.DataFrame,
    score_fn,
    alphas: list[float] = tuple(ALPHA_SWEEP),
    pool_solutions: int = POOL_SOLUTIONS,
) -> pd.DataFrame:
    rows = []
    for a in alphas:
        print(f"  [sweep] alpha = {a:.1e}")
        vndf_a = CalcBounds(vndf_base.copy(), a)
        vndf_a = AdjustEnfs(vndf_a, dndf)
        res = solve_and_select(
            vndf_a, vndf_base, dndf,
            pool_solutions=pool_solutions,
            seed=MIP_SEED,
            score_fn=score_fn,
        )
        row = {
            "alpha": a,
            "feasible": bool(res.feasible),
            "n_constraints": int(res.n_constraints),
            "pool_size": int(res.pool_size),
            "mip_gap": res.mip_gap,
            "best_d_obj": (float(res.d_obj[res.best_d_idx]) if res.feasible else None),
            "n_at_best_d": int(res.n_at_best_d) if res.feasible else 0,
            "best_v_ll": (float(res.v_ll[res.best_v_idx]) if res.feasible else None),
            "n_at_best_v": int(res.n_at_best_v) if res.feasible else 0,
            "argmax_agree": bool(res.best_d_idx == res.best_v_idx) if res.feasible else None,
            "pred_hamming_d_vs_v": (int((res.pred_d != res.pred_v).sum()) if res.feasible else None),
            "spearman_rho": res.spearman_rho,
            "runtime_s": round(res.runtime, 2),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def pick_alpha_star(sweep_df: pd.DataFrame) -> float:
    """Largest alpha for which the MIP is feasible."""
    feasible = sweep_df[sweep_df["feasible"]]
    if feasible.empty:
        raise RuntimeError("No feasible alpha found in sweep")
    return float(feasible["alpha"].max())


# ──────────────────────────────────────────────────────────────────────────
# Full alpha* solve: produce both selector predictions
# ──────────────────────────────────────────────────────────────────────────

def solve_alpha_star(
    dndf: pd.DataFrame,
    vndf_base: pd.DataFrame,
    score_fn,
    alpha_star: float,
    pool_solutions: int = POOL_SOLUTIONS,
):
    print(f"  [alpha*] CalcBounds + AdjustEnfs at alpha = {alpha_star:.1e}")
    vndf_star = CalcBounds(vndf_base.copy(), alpha_star)
    vndf_star = AdjustEnfs(vndf_star, dndf)
    print(f"  [alpha*] solve_and_select (pool={pool_solutions})")
    res = solve_and_select(
        vndf_star, vndf_base, dndf,
        pool_solutions=pool_solutions,
        seed=MIP_SEED,
        score_fn=score_fn,
    )
    if not res.feasible:
        raise RuntimeError(f"Infeasible at alpha*={alpha_star}")
    return vndf_star, res


# ──────────────────────────────────────────────────────────────────────────
# Build per-node analysis tables
# ──────────────────────────────────────────────────────────────────────────

def build_dnode_table(
    dndf: pd.DataFrame,
    pred_d: np.ndarray,
    pred_v: np.ndarray,
    baseline_res: pd.DataFrame,
) -> pd.DataFrame:
    """Merge FB (both selectors) with baselines into one d-node table."""
    out = dndf.copy()
    out["Pred_FB_d"] = pred_d.astype(int)
    out["Pred_FB_v"] = pred_v.astype(int)
    out["N_Pred_FB_d"] = out["Pred_FB_d"] * out["Count"]
    out["N_Pred_FB_v"] = out["Pred_FB_v"] * out["Count"]

    # Merge baseline predictions on the full feature tuple.
    # dndf stores m, M as object columns; baseline_res has only feature+Count+preds.
    feature_cols = list(
        dndf.columns[: dndf.columns.get_loc("Count")]
    )
    # dtype coercion: dndf feature values are strings/ints; baselines likewise
    # but from value_counts — should match natively.
    b = baseline_res[feature_cols + ["DT_Pred", "NN_Pred", "MC_Pred"]].copy()
    out = out.merge(b, on=feature_cols, how="left")
    for col in ("DT_Pred", "NN_Pred", "MC_Pred"):
        out[col] = out[col].fillna(0).astype(int)
    return out


def build_vnode_table(
    vndf_base: pd.DataFrame,
    vndf_star: pd.DataFrame,
    dndf_with_preds: pd.DataFrame,
) -> pd.DataFrame:
    """Augment vndf_base with aggregated predictions per model + Vmin/Vmax at alpha*.

    For each v-node v and model M the aggregated prediction is

        N_Pred_M_v  =  Σ_{i ∈ dchildren(v)}  N_{d_i} · x_M_i
                    =  # individuals M predicts positive inside v
                    =  P_v_M   (the paper's v-node prediction quantity).

    x_M_i ∈ {0,1} is M's binary d-node decision (all four models are
    d-node-deterministic: same features ⇒ same label, so each d-node's
    predicted-positive count is always 0 or Count_{d_i}).

    Consistent_M_v is true iff Vmin(v) ≤ N_Pred_M_v ≤ Vmax(v). This is
    exactly the constraint the FB MIP enforces (after the April 2026 fix
    swapping Target for Count coefficients).
    """
    cat_cols = list(vndf_base.columns[: vndf_base.columns.get_loc("Count")])

    # Per-alpha* Vmin/Vmax lookup (v-nodes pruned by AdjustEnfs get [0, Count]).
    star_map: dict[tuple, tuple[int, int]] = {}
    for _, row in vndf_star.iterrows():
        k = tuple(str(row[c]) for c in cat_cols)
        star_map[k] = (int(row["Vmin"]), int(row["Vmax"]))

    cats = vndf_base[cat_cols].astype(str)
    vnode_keys = list(map(tuple, cats.values.tolist()))

    vmin_col = np.zeros(len(vndf_base), dtype=int)
    vmax_col = np.zeros(len(vndf_base), dtype=int)
    active_col = np.zeros(len(vndf_base), dtype=bool)
    counts = vndf_base["Count"].to_numpy()
    for i, k in enumerate(vnode_keys):
        if k in star_map:
            vmin_col[i], vmax_col[i] = star_map[k]
            active_col[i] = True
        else:
            vmin_col[i], vmax_col[i] = 0, int(counts[i])
            active_col[i] = False

    out = vndf_base.copy()
    out["Vmin"] = vmin_col
    out["Vmax"] = vmax_col
    out["ActiveConstraint"] = active_col

    count_d = dndf_with_preds["Count"].to_numpy()

    # Binary d-node decision per model. For FB these are already 0/1. For
    # baselines the per-d-node prediction count is always 0 or Count_i, so
    # (col > 0) recovers the binary.
    binary_cols = {
        "FB_d": dndf_with_preds["Pred_FB_d"].to_numpy().astype(int),
        "FB_v": dndf_with_preds["Pred_FB_v"].to_numpy().astype(int),
        "DT":   (dndf_with_preds["DT_Pred"].to_numpy() > 0).astype(int),
        "NN":   (dndf_with_preds["NN_Pred"].to_numpy() > 0).astype(int),
        "MC":   (dndf_with_preds["MC_Pred"].to_numpy() > 0).astype(int),
    }

    dchildren = vndf_base["dchildren"].to_numpy()

    for m_label, bin_vec in binary_cols.items():
        n_per_d = (count_d * bin_vec).astype(int)
        n_agg = np.zeros(len(vndf_base), dtype=int)
        for i, dc in enumerate(dchildren):
            idx = np.asarray(dc, dtype=int)
            n_agg[i] = int(n_per_d[idx].sum())
        out[f"N_Pred_{m_label}"] = n_agg
        out[f"Consistent_{m_label}"] = (n_agg >= vmin_col) & (n_agg <= vmax_col)

    return out


# ──────────────────────────────────────────────────────────────────────────
# Paper tables
# ──────────────────────────────────────────────────────────────────────────

def table_node_counts(data: pd.DataFrame, dndf: pd.DataFrame, vndf_base: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "n_entries": len(data),
        "n_d_nodes": len(dndf),
        "n_v_nodes": len(vndf_base),
    }])


def table_d01_error(data_n: int, dndf_with_preds: pd.DataFrame) -> pd.DataFrame:
    """Proportion of instances assigned a prediction inconsistent with their d_0/d_1 node.

    For d_0 nodes (Vmax < Count, only consistent deterministic label = 0):
        # inconsistent = # predicted positive = Pred × Count
    For d_1 nodes (Vmin > 0, only consistent deterministic label = 1):
        # inconsistent = # predicted negative = (1 - Pred) × Count
    FB satisfies by construction (Pred_FB_d and Pred_FB_v agree with E_Cat).
    """
    rows = []
    for model in ("FB_d", "FB_v", "DT", "NN", "MC"):
        col = (f"Pred_{model}" if model in ("FB_d", "FB_v") else f"{model}_Pred")
        pred = dndf_with_preds[col].to_numpy()
        count = dndf_with_preds["Count"].to_numpy()
        e_cat = dndf_with_preds["E_Cat"].to_numpy()
        err = 0
        # Per-instance errors: baseline's per-instance predictions aren't stored,
        # but for DT/NN/PMC the aggregate "N_Pred" = # predicted positive in d-node.
        # For a d_0 node, error = # predicted positive; for d_1, error = # negative.
        for i in range(len(dndf_with_preds)):
            if e_cat[i] == "E0":
                if model in ("FB_d", "FB_v"):
                    err += int(pred[i]) * int(count[i])
                else:
                    # # predicted positive aggregated from baseline
                    err += int(dndf_with_preds[col].iloc[i])
            elif e_cat[i] == "E1":
                if model in ("FB_d", "FB_v"):
                    err += (1 - int(pred[i])) * int(count[i])
                else:
                    err += int(count[i]) - int(dndf_with_preds[col].iloc[i])
        rows.append({"model": model, "n_inconsistent": int(err), "pct_inconsistent": round(100 * err / data_n, 2)})
    return pd.DataFrame(rows)


def table_dnf_summary(data_n: int, dndf: pd.DataFrame) -> pd.DataFrame:
    enf = dndf[dndf["E_Cat"] == "Enf"]
    return pd.DataFrame([{
        "total_instances": data_n,
        "instances_in_dnf": int(enf["Count"].sum()),
        "pct_in_dnf": round(100 * enf["Count"].sum() / data_n, 1),
        "n_dnf_nodes": len(enf),
    }])


def table_vnode_error(vndf_with_preds: pd.DataFrame) -> pd.DataFrame:
    """For active v-node constraints at alpha*, fraction of v-nodes whose
    aggregate prediction falls outside [Vmin, Vmax], per model.
    """
    active = vndf_with_preds[vndf_with_preds["ActiveConstraint"]]
    n_active = len(active)
    rows = []
    for m in ("FB_d", "FB_v", "DT", "NN", "MC"):
        inc = int((~active[f"Consistent_{m}"]).sum())
        rows.append({
            "model": m,
            "n_active_v_nodes": int(n_active),
            "n_inconsistent": inc,
            "pct_inconsistent": (round(100 * inc / n_active, 2) if n_active else None),
        })
    return pd.DataFrame(rows)


def table_accuracy(dndf_with_preds: pd.DataFrame) -> pd.DataFrame:
    """Overall accuracy omitting d_nf nodes. Accuracy for FB uses both selectors.

    Accuracy = 1 if predicted == true label for every instance, summed.
        correct_M = Σ_{i not in d_nf} [Pred_M_i * Target_i + (1 - Pred_M_i) * (Count_i - Target_i)]
    For baselines where we have # predicted positive (not per-instance),
    the natural aggregate is:
        # correct = min(N_Pred_M, Target) + min(Count - N_Pred_M, Count - Target)
    but that's a generous upper bound. Instead we use the per-d-node accuracy
    as if predictions are split evenly — the existing FB4 notebook uses the
    FB assumption (where Pred is uniform across d-node individuals):
        correct = Pred * Target + (1 - Pred) * (Count - Target)  for FB
    For baselines, we can't recover per-instance labels from aggregated dnode
    counts without recomputing — but since y_pred_dt etc. were stored, the
    simpler path is: accuracy = Σ |Pred_M_dnode_positives - Target|-adjusted.
    To avoid re-running baselines, we use the notebook's definition:
        acc(pred_col) = 100 * Σ np.where(pred > 0, Target, Count - Target) / Σ Count
    which treats Pred_col as a per-d-node binary decision. For DT/NN/PMC this
    reflects the majority vote within d-node (Pred_col = aggregated_positive).

    This matches the current paper's numbers.
    """
    no_enf = dndf_with_preds[dndf_with_preds["E_Cat"] != "Enf"]
    total = int(no_enf["Count"].sum())

    def acc_fb(col):
        pred = no_enf[col].to_numpy()  # binary 0/1
        correct = np.where(
            pred > 0, no_enf["Target"].to_numpy(), no_enf["Count"].to_numpy() - no_enf["Target"].to_numpy()
        ).sum()
        return round(100 * correct / total, 2)

    def acc_baseline(col):
        pred_positives = no_enf[col].to_numpy()  # aggregated count
        # Per-instance accuracy = min(pred_pos, target) + min(count-pred_pos, count-target)
        target = no_enf["Target"].to_numpy()
        count = no_enf["Count"].to_numpy()
        correct = np.minimum(pred_positives, target) + np.minimum(count - pred_positives, count - target)
        return round(100 * correct.sum() / total, 2)

    return pd.DataFrame([{
        "FB_d": acc_fb("Pred_FB_d"),
        "FB_v": acc_fb("Pred_FB_v"),
        "DT": acc_baseline("DT_Pred"),
        "NN": acc_baseline("NN_Pred"),
        "MC": acc_baseline("MC_Pred"),
    }])


def table_multical(dndf_with_preds: pd.DataFrame, protected_vars: list[str]) -> pd.DataFrame:
    """Per-group (predicted rate - observed rate) per model, omitting d_nf."""
    no_enf = dndf_with_preds[dndf_with_preds["E_Cat"] != "Enf"].copy()
    no_enf["N_Pred_FB_d"] = no_enf["Pred_FB_d"] * no_enf["Count"]
    no_enf["N_Pred_FB_v"] = no_enf["Pred_FB_v"] * no_enf["Count"]

    grp = (
        no_enf
        .groupby(protected_vars)
        .agg(
            Count=("Count", "sum"),
            Target=("Target", "sum"),
            N_FB_d=("N_Pred_FB_d", "sum"),
            N_FB_v=("N_Pred_FB_v", "sum"),
            N_DT=("DT_Pred", "sum"),
            N_NN=("NN_Pred", "sum"),
            N_MC=("MC_Pred", "sum"),
        )
        .reset_index()
    )
    grp["Observed Rate"] = (grp["Target"] / grp["Count"]).round(3)
    for m, col in [("FB_d", "N_FB_d"), ("FB_v", "N_FB_v"), ("DT", "N_DT"), ("NN", "N_NN"), ("MC", "N_MC")]:
        grp[f"{m} Rate"] = (grp[col] / grp["Count"]).round(3)
        grp[f"{m} MC Err"] = (grp[f"{m} Rate"] - grp["Observed Rate"]).round(3)
    keep = protected_vars + ["Count", "Target", "Observed Rate"] + [
        f"{m} Rate" for m in ("FB_d", "FB_v", "DT", "NN", "MC")
    ] + [f"{m} MC Err" for m in ("FB_d", "FB_v", "DT", "NN", "MC")]
    return grp[keep]


# ──────────────────────────────────────────────────────────────────────────
# Illustrative-example candidates
# ──────────────────────────────────────────────────────────────────────────

def _feature_cols(dndf: pd.DataFrame) -> list[str]:
    return list(dndf.columns[: dndf.columns.get_loc("Count")])


def candidates_d01_violation(dndf_with_preds: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
    """Rank d_0 and d_1 nodes by baseline violation magnitude.

    For d_0 nodes: violation = max over baselines of N_Pred_M (should be ~0).
    For d_1 nodes: violation = max over baselines of (Count - N_Pred_M).
    """
    df = dndf_with_preds.copy()
    df = df[(df["E_Cat"].isin(("E0", "E1"))) & (df["Count"] >= min_count)].copy()
    if df.empty:
        return df

    def row_violation(row):
        count = int(row["Count"])
        baseline_pos = np.array([int(row["DT_Pred"]), int(row["NN_Pred"]), int(row["MC_Pred"])])
        if row["E_Cat"] == "E0":
            v = int(baseline_pos.max())
            worst = ["DT", "NN", "MC"][int(np.argmax(baseline_pos))]
        else:
            neg = count - baseline_pos
            v = int(neg.max())
            worst = ["DT", "NN", "MC"][int(np.argmax(neg))]
        return pd.Series({"violation": v, "worst_model": worst})

    metrics = df.apply(row_violation, axis=1)
    df = pd.concat([df, metrics], axis=1)
    df = df[df["violation"] > 0]
    if df.empty:
        return df
    df = df.sort_values(["violation", "Count"], ascending=[False, False]).reset_index(drop=True)

    keep_cols = (
        _feature_cols(dndf_with_preds)
        + [
            "Count", "Target", "Vmin", "Vmax", "E_Cat",
            "Pred_FB_d", "Pred_FB_v",
            "DT_Pred", "NN_Pred", "MC_Pred",
            "violation", "worst_model",
        ]
    )
    return df[keep_cols]


def candidates_dnf_abstention(dndf_with_preds: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
    """d_nf nodes with target rate clearly in the middle (both labels inconsistent)."""
    df = dndf_with_preds[
        (dndf_with_preds["E_Cat"] == "Enf") & (dndf_with_preds["Count"] >= min_count)
    ].copy()
    if df.empty:
        return df
    df["observed_rate"] = (df["Target"] / df["Count"]).round(3)
    df["midness"] = (0.5 - (df["observed_rate"] - 0.5).abs()).round(3)
    df = df.sort_values(["midness", "Count"], ascending=[False, False]).reset_index(drop=True)

    keep_cols = (
        _feature_cols(dndf_with_preds)
        + [
            "Count", "Target", "observed_rate", "midness",
            "Vmin", "Vmax",
            "DT_Pred", "NN_Pred", "MC_Pred",
        ]
    )
    return df[keep_cols]


def candidates_vnode_inconsistency(
    vndf_with_preds: pd.DataFrame,
    min_count: int = 50,
) -> pd.DataFrame:
    """v-nodes where a baseline's N_Pred (paper P_v) is outside [Vmin, Vmax].

    Ranked by violation magnitude (= max over baselines of max(N - Vmax, Vmin - N)).
    The paper's illustrative v-node (Adult FM2560) compares the column
    reported here (# individuals predicted positive) against [Vmin, Vmax].
    """
    df = vndf_with_preds[
        vndf_with_preds["ActiveConstraint"] & (vndf_with_preds["Count"] >= min_count)
    ].copy()
    if df.empty:
        return df

    def compute_violation(row):
        lo, hi = int(row["Vmin"]), int(row["Vmax"])
        best_v = 0
        worst = ""
        for m in ("DT", "NN", "MC"):
            n = int(row[f"N_Pred_{m}"])
            v = max(n - hi, lo - n, 0)
            if v > best_v:
                best_v = v
                worst = m
        return pd.Series({"violation": best_v, "worst_model": worst})

    metrics = df.apply(compute_violation, axis=1)
    df = pd.concat([df, metrics], axis=1)
    df = df[df["violation"] > 0]
    if df.empty:
        return df
    df = df.sort_values(["violation", "Count"], ascending=[False, False]).reset_index(drop=True)

    cat_cols = list(vndf_with_preds.columns[: vndf_with_preds.columns.get_loc("Count")])
    keep_cols = (
        cat_cols
        + [
            "Count", "Target", "Vmin", "Vmax",
            "N_Pred_FB_d", "N_Pred_FB_v",
            "N_Pred_DT", "N_Pred_NN", "N_Pred_MC",
            "violation", "worst_model",
        ]
    )
    return df[keep_cols]


# ──────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────

def plot_dnode_sizes(dndf: pd.DataFrame, ds: str, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sizes = np.sort(dndf["Count"].values)[::-1]
    ranks = np.arange(1, len(sizes) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    baseline = 0.5
    ax.vlines(ranks, baseline, sizes, color="steelblue", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("d-node index")
    ax.set_ylabel("d-node size (log scale)")
    ax.set_xlim(-0.02 * len(ranks), len(ranks) * 1.02)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_ylim(baseline, None)
    plt.tight_layout()
    out = out_dir / f"{ds}_dnodes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_race_dnodes(dndf: pd.DataFrame, ds: str, out_dir: Path, race_col: str = "race_cat") -> Path | None:
    if race_col not in dndf.columns:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size_edges = [0, 25, 50, 100, float("inf")]
    size_labels = ["1-25", "26-50", "51-100", ">100"]
    race_groups = sorted(dndf[race_col].unique().tolist())
    # Monochrome-safe sequential blues (matplotlib `Blues` colormap).
    # Distinct luminance levels remain distinguishable when printed in B&W
    # and stay consistent with `steelblue` used elsewhere in the paper.
    _BLUES = ["#08306b", "#9ecae1", "#4292c6", "#c6dbef"]
    colors = _BLUES[: len(race_groups)]

    local = dndf[[race_col, "Count"]].copy()
    local["_rb"] = pd.cut(local["Count"], bins=size_edges, labels=size_labels)
    props = {}
    for g in race_groups:
        sub = local[local[race_col] == g]
        total_ind = sub["Count"].sum()
        props[g] = (
            sub.groupby("_rb", observed=True)["Count"].sum().reindex(size_labels, fill_value=0)
            / total_ind
        ).values

    x, w = np.arange(len(size_labels)), 0.8 / len(race_groups)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (g, c) in enumerate(zip(race_groups, colors)):
        offset = (i - (len(race_groups) - 1) / 2) * w
        bars = ax.bar(
            x + offset, props[g] * 100, w,
            label=str(g).capitalize(), color=c, alpha=0.85, edgecolor="white",
        )
        for bar, pct in zip(bars, props[g] * 100):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.set_xlabel("d-node size")
    ax.set_ylabel("Proportion of individuals (%)")
    ax.set_title("Proportion of Individuals by d-node Size Bin and Race")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    out = out_dir / f"{ds}_race_dnodes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Report assembly
# ──────────────────────────────────────────────────────────────────────────

def _fmt_df(
    df: pd.DataFrame,
    floatfmt: str = ".3f",
    maxrows: int | None = None,
    disable_numparse: bool = False,
) -> str:
    from tabulate import tabulate
    if maxrows is not None and len(df) > maxrows:
        df = df.head(maxrows)
    return tabulate(
        df,
        headers="keys",
        tablefmt="pipe",
        floatfmt=floatfmt,
        showindex=False,
        disable_numparse=disable_numparse,
    )


def write_report(ds: str, ds_dir: Path, summary: dict, tables: dict[str, pd.DataFrame], candidates: dict[str, pd.DataFrame]):
    lines = [f"# {ds.upper()} — FairBayesian results\n"]

    # Header summary
    lines.append("## Summary\n")
    lines.append(f"- Dataset: **{ds}**  ({summary['n_entries']:,} instances)")
    lines.append(f"- d-nodes: **{summary['n_d_nodes']:,}**   v-nodes: **{summary['n_v_nodes']:,}**")
    lines.append(f"- alpha*: **{summary['alpha_star']:.1e}**   pool: **{summary['pool_solutions']}**")
    lines.append(
        f"- MIP gap: **{summary['alpha_star_mip_gap']}**   "
        f"N@Best (D-obj): **{summary['alpha_star_n_at_best_d']}**   "
        f"N@Best (V-LL): **{summary['alpha_star_n_at_best_v']}**"
    )
    if summary.get("alpha_star_argmax_agree") is False:
        lines.append(
            f"- ⚠ D-best and V-best **disagree** at alpha*: "
            f"Hamming = {summary['alpha_star_hamming_d_vs_v']}"
        )
    else:
        lines.append("- D-best and V-best **agree** at alpha*.")
    lines.append("")

    lines.append("## Node counts\n")
    lines.append(_fmt_df(tables["node_counts"], floatfmt=".0f"))
    lines.append("")

    lines.append("## Alpha sweep\n")
    sweep_disp = tables["alpha_sweep"].copy()
    if "alpha" in sweep_disp.columns:
        sweep_disp["alpha"] = sweep_disp["alpha"].map(
            lambda a: "-" if pd.isna(a) else f"{a:.1e}"
        )
    # Round other numeric columns separately then render with numparse off so
    # the pre-formatted alpha strings stay as-is.
    for col in sweep_disp.select_dtypes(include="float").columns:
        sweep_disp[col] = sweep_disp[col].map(
            lambda x: "-" if pd.isna(x) else f"{x:.3f}"
        )
    lines.append(_fmt_df(sweep_disp, disable_numparse=True))
    lines.append("")

    lines.append("## d_{0/1} node consistency error\n")
    lines.append(_fmt_df(tables["d01_error"], floatfmt=".2f"))
    lines.append("")

    lines.append("## d_nf summary\n")
    lines.append(_fmt_df(tables["dnf_summary"], floatfmt=".1f"))
    lines.append("")

    lines.append("## v-node consistency error\n")
    lines.append(
        "Active v-node constraints at alpha* where "
        "Vmin(v) ≤ Σ N_d · x_d ≤ Vmax(v) fails, per model."
    )
    lines.append("")
    lines.append(_fmt_df(tables["vnode_error"], floatfmt=".2f"))
    lines.append("")

    lines.append("## Accuracy (omitting d_nf)\n")
    lines.append(_fmt_df(tables["accuracy"], floatfmt=".2f"))
    lines.append("")

    lines.append("## Multicalibration (per protected-group cell)\n")
    lines.append(_fmt_df(tables["multical"], floatfmt=".3f"))
    lines.append("")

    lines.append("## Illustrative-example candidates\n")
    for key, title in [
        ("d01_violation", "d_{0/1} violation candidates (top 20)"),
        ("dnf_abstention", "d_nf abstention candidates (top 20)"),
        ("vnode_inconsistency", "v-node inconsistency candidates (top 20)"),
    ]:
        df = candidates.get(key)
        lines.append(f"### {title}\n")
        if df is None or df.empty:
            lines.append("_(no candidates found at current filters)_")
        else:
            lines.append(_fmt_df(df.head(20), floatfmt=".3f"))
        lines.append("")

    lines.append("## Files\n")
    lines.append(f"- `{ds}/summary.json`  — machine-readable summary")
    lines.append(f"- `{ds}/dnodes.parquet`  — per-d-node table with all five prediction columns")
    lines.append(f"- `{ds}/vnodes.parquet`  — per-v-node table with aggregated prediction counts & consistency booleans")
    lines.append(f"- `{ds}/tables/tab_*.csv`  — one CSV per paper table")
    lines.append(f"- `{ds}/examples/candidates_*.csv`  — full ranked candidate lists")
    lines.append(f"- `{ds}/figures/*.png`  — d-node size and race breakdown")
    lines.append("")

    (ds_dir / "report.md").write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────
# Dataset-level driver
# ──────────────────────────────────────────────────────────────────────────

def run_one(
    ds: str,
    results_root: Path,
    do_alpha_sweep: bool = True,
    skip_baselines: bool = False,
    skip_pmc: bool = False,
    force: bool = False,
) -> dict:
    print(f"\n========================  {ds.upper()}  ========================")
    t_ds = time.time()
    data, dndf, vndf_base, cfg, ds_dir = prepare_dataset(ds, results_root, force=force)
    data_n = len(data)
    protected_vars = cfg["protected_vars"]

    _ensure_dir(ds_dir / "tables")
    _ensure_dir(ds_dir / "examples")
    _ensure_dir(ds_dir / "figures")

    # V-LL scorer (reused everywhere)
    with _stage(f"[{ds}] build V-LL scorer"):
        score_fn, _, _ = build_vll_scorer(vndf_base, dndf)

    # Baselines (cached)
    if skip_baselines:
        baseline_res = pd.DataFrame(columns=_feature_cols(dndf) + ["Count", "DT_Pred", "NN_Pred", "MC_Pred"])
        baseline_meta = {"skipped": True}
    else:
        baseline_res, baseline_meta = cache_or_run_baselines(
            data, protected_vars, ds_dir, force=force, skip_pmc=skip_pmc,
        )

    # Alpha sweep
    sweep_df = None
    if do_alpha_sweep:
        with _stage(f"[{ds}] alpha sweep"):
            sweep_df = alpha_sweep(dndf, vndf_base, score_fn)
        sweep_df.to_csv(ds_dir / "tables" / "tab_alpha_sweep.csv", index=False)
        alpha_star = pick_alpha_star(sweep_df)
    else:
        alpha_star = float(cfg["alpha_star"])

    print(f"[{ds}] using alpha* = {alpha_star:.1e}")

    # Alpha* solve (full pool, both selectors)
    with _stage(f"[{ds}] alpha* solve"):
        vndf_star, res = solve_alpha_star(dndf, vndf_base, score_fn, alpha_star)

    # Node-level tables
    with _stage(f"[{ds}] build dnodes + vnodes"):
        dnode_tbl = build_dnode_table(dndf, res.pred_d, res.pred_v, baseline_res)
        vnode_tbl = build_vnode_table(vndf_base, vndf_star, dnode_tbl)
    dnode_tbl.drop(columns=["m", "M"], errors="ignore").to_parquet(ds_dir / "dnodes.parquet")
    # vnode feature columns are a mix of strings ("Male") and int -1 (ambiguous).
    # parquet can't store ragged object arrays, so drop dchildren and cast
    # every object-dtype column to string.
    vnode_export = vnode_tbl.drop(columns=["m", "M", "dchildren"], errors="ignore").copy()
    for col in vnode_export.select_dtypes(include="object").columns:
        vnode_export[col] = vnode_export[col].astype(str)
    vnode_export.to_parquet(ds_dir / "vnodes.parquet")

    # Paper tables
    with _stage(f"[{ds}] paper tables"):
        tables = {
            "node_counts": table_node_counts(data, dndf, vndf_base),
            "d01_error": table_d01_error(data_n, dnode_tbl),
            "dnf_summary": table_dnf_summary(data_n, dndf),
            "vnode_error": table_vnode_error(vnode_tbl),
            "accuracy": table_accuracy(dnode_tbl),
            "multical": table_multical(dnode_tbl, protected_vars),
        }
        if sweep_df is not None:
            tables["alpha_sweep"] = sweep_df
        else:
            # Reuse existing sweep CSV if present, else write a placeholder so
            # the table list is complete.
            existing = ds_dir / "tables" / "tab_alpha_sweep.csv"
            if existing.exists():
                tables["alpha_sweep"] = pd.read_csv(existing)
            else:
                tables["alpha_sweep"] = pd.DataFrame([{"note": "skipped (--no-alpha-sweep)"}])
        for name, df in tables.items():
            if name == "alpha_sweep" and sweep_df is None and (ds_dir / "tables" / "tab_alpha_sweep.csv").exists():
                # Don't rewrite the existing sweep CSV when skipping the sweep.
                continue
            df.to_csv(ds_dir / "tables" / f"tab_{name}.csv", index=False)

    # Illustrative candidates
    with _stage(f"[{ds}] illustrative candidates"):
        candidates = {
            "d01_violation": candidates_d01_violation(dnode_tbl),
            "dnf_abstention": candidates_dnf_abstention(dnode_tbl),
            "vnode_inconsistency": candidates_vnode_inconsistency(vnode_tbl),
        }
        for name, df in candidates.items():
            df.to_csv(ds_dir / "examples" / f"candidates_{name}.csv", index=False)

    # Figures
    with _stage(f"[{ds}] figures"):
        plot_dnode_sizes(dndf, ds, ds_dir / "figures")
        plot_race_dnodes(dndf, ds, ds_dir / "figures")

    # Summary
    summary = {
        "dataset": ds,
        "protected_vars": protected_vars,
        "n_entries": data_n,
        "n_d_nodes": len(dndf),
        "n_v_nodes": len(vndf_base),
        "pool_solutions": POOL_SOLUTIONS,
        "alpha_star": alpha_star,
        "alpha_star_mip_gap": res.mip_gap,
        "alpha_star_pool_size": res.pool_size,
        "alpha_star_best_d_obj": float(res.d_obj[res.best_d_idx]),
        "alpha_star_best_v_ll": float(res.v_ll[res.best_v_idx]),
        "alpha_star_n_at_best_d": res.n_at_best_d,
        "alpha_star_n_at_best_v": res.n_at_best_v,
        "alpha_star_argmax_agree": bool(res.best_d_idx == res.best_v_idx),
        "alpha_star_hamming_d_vs_v": int((res.pred_d != res.pred_v).sum()),
        "alpha_star_spearman_rho": res.spearman_rho,
        "d_cat_counts": dndf["E_Cat"].value_counts().to_dict(),
        "baseline_meta": baseline_meta,
        "runtime_s": round(time.time() - t_ds, 2),
    }
    (ds_dir / "summary.json").write_text(json.dumps(_to_json_safe(summary), indent=2))

    # Report
    with _stage(f"[{ds}] report.md"):
        write_report(ds, ds_dir, summary, tables, candidates)

    return summary


# ──────────────────────────────────────────────────────────────────────────
# Cross-dataset roll-up
# ──────────────────────────────────────────────────────────────────────────

def rollup(summaries: list[dict], results_root: Path):
    """Write a cross-dataset report.md + combined CSVs.

    Pulls v-node error and accuracy tables per dataset from disk so the
    rollup stays consistent with the per-dataset reports.
    """
    # Per-dataset headline summary. Numbers that tabulate would reformat as
    # scientific or decimals are cast to strings so the report renders nicely.
    headline = pd.DataFrame([
        {
            "dataset": s["dataset"],
            "n_entries": f"{s['n_entries']:,}",
            "n_d_nodes": f"{s['n_d_nodes']:,}",
            "n_v_nodes": f"{s['n_v_nodes']:,}",
            "pct_in_dnf": None,
            "alpha_star": f"{s['alpha_star']:.1e}",
            "n_at_best_d": s["alpha_star_n_at_best_d"],
            "n_at_best_v": s["alpha_star_n_at_best_v"],
            "D==V": "yes" if s["alpha_star_argmax_agree"] else "no",
            "spearman_rho": f"{s['alpha_star_spearman_rho']:.3f}",
        }
        for s in summaries
    ])

    # Reach into per-dataset tables for richer cross-dataset comparisons.
    accuracy_rows: list[dict] = []
    vnode_err_rows: list[dict] = []
    d01_err_rows: list[dict] = []
    dnf_rows: list[dict] = []
    for s in summaries:
        ds = s["dataset"]
        ds_tab = results_root / ds / "tables"
        if (ds_tab / "tab_accuracy.csv").exists():
            acc = pd.read_csv(ds_tab / "tab_accuracy.csv").iloc[0].to_dict()
            acc["dataset"] = ds
            accuracy_rows.append(acc)
        if (ds_tab / "tab_vnode_error.csv").exists():
            for _, row in pd.read_csv(ds_tab / "tab_vnode_error.csv").iterrows():
                r = row.to_dict()
                r["dataset"] = ds
                vnode_err_rows.append(r)
        if (ds_tab / "tab_d01_error.csv").exists():
            for _, row in pd.read_csv(ds_tab / "tab_d01_error.csv").iterrows():
                r = row.to_dict()
                r["dataset"] = ds
                d01_err_rows.append(r)
        if (ds_tab / "tab_dnf_summary.csv").exists():
            d = pd.read_csv(ds_tab / "tab_dnf_summary.csv").iloc[0].to_dict()
            d["dataset"] = ds
            dnf_rows.append(d)
            # patch pct_in_dnf into headline
            headline.loc[headline["dataset"] == ds, "pct_in_dnf"] = f"{d['pct_in_dnf']:.1f}%"

    accuracy_all = pd.DataFrame(accuracy_rows)[["dataset", "FB_d", "FB_v", "DT", "NN", "MC"]]
    d01_all = pd.DataFrame(d01_err_rows).pivot(
        index="dataset", columns="model", values="pct_inconsistent"
    ).reindex(columns=["FB_d", "FB_v", "DT", "NN", "MC"]).reset_index()
    dnf_all = pd.DataFrame(dnf_rows)[["dataset", "total_instances", "instances_in_dnf",
                                      "pct_in_dnf", "n_dnf_nodes"]]

    vnode_err_all = pd.DataFrame(vnode_err_rows)
    vnode_pct = vnode_err_all.pivot(
        index="dataset", columns="model", values="pct_inconsistent"
    ).reindex(columns=["FB_d", "FB_v", "DT", "NN", "MC"]).reset_index()

    accuracy_all.to_csv(results_root / "tab_accuracy_all.csv", index=False)
    d01_all.to_csv(results_root / "tab_d01_pct_all.csv", index=False)
    dnf_all.to_csv(results_root / "tab_dnf_summary_all.csv", index=False)
    vnode_pct.to_csv(results_root / "tab_vnode_pct_all.csv", index=False)
    headline.to_csv(results_root / "tab_headline_all.csv", index=False)

    lines: list[str] = [
        "# FairBayesian JAIR results — all datasets\n",
        "## Headline\n",
        _fmt_df(headline, disable_numparse=True),
        "",
        "## Accuracy (%, omitting d_nf)\n",
        _fmt_df(accuracy_all, floatfmt=".2f"),
        "",
        "## d_{0/1} consistency error (%)\n",
        _fmt_df(d01_all, floatfmt=".2f"),
        "",
        "## d_nf summary\n",
        _fmt_df(dnf_all, floatfmt=".1f"),
        "",
        "## v-node consistency error (%)  —  Vmin(v) ≤ Σ N_d · x_d ≤ Vmax(v)\n",
        _fmt_df(vnode_pct, floatfmt=".2f"),
        "",
    ]
    (results_root / "report.md").write_text("\n".join(lines))

    combined_summary = {s["dataset"]: s for s in summaries}
    (results_root / "summary_all.json").write_text(json.dumps(_to_json_safe(combined_summary), indent=2))


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["adult", "compas", "bm", "all"], default="all")
    parser.add_argument("--no-alpha-sweep", dest="alpha_sweep", action="store_false")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-pmc", action="store_true",
                        help="train DT+NN but skip the PMC baseline (PMC fit can be slow)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild CalcMs and baseline caches")
    parser.add_argument("--skip-verify", action="store_true",
                        help="don't require a prior uniqueness verification")
    parser.add_argument("--results-root", default=str(PHD_DIR / "results"))
    args = parser.parse_args()

    results_root = Path(args.results_root)
    _ensure_dir(results_root)

    datasets = ["adult", "compas", "bm"] if args.dataset == "all" else [args.dataset]

    # Pre-flight: require an uniqueness check report per dataset
    if not args.skip_verify:
        missing = []
        for ds in datasets:
            check = results_root / ds / "uniqueness_check.json"
            if not check.exists():
                missing.append(str(check))
        if missing:
            print("⚠  Uniqueness checks not yet run for:")
            for m in missing:
                print(f"   - {m}")
            print("   Run `python verify_uniqueness.py` first, or pass --skip-verify to bypass.")
            sys.exit(1)

    summaries = []
    for ds in datasets:
        summary = run_one(
            ds,
            results_root,
            do_alpha_sweep=args.alpha_sweep,
            skip_baselines=args.skip_baselines,
            skip_pmc=args.skip_pmc,
            force=args.force,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        rollup(summaries, results_root)

    print("\n✓ run_experiments done.")


if __name__ == "__main__":
    main()
