"""
Pre-flight uniqueness verification for the FairBayesian MIP.

Runs a small set of checks to confirm:
  1. The MIP reaches a global optimum (MIPGap == 0).
  2. The top solution by D-obj is unique (N@Best_d == 1) at alpha*.
  3. The top solution by V-LL is unique (N@Best_v == 1) at alpha*.
  4. D-best and V-best coincide at alpha* (else we have a live disagreement).
  5. The V-best solution is stable as pool_solutions grows (100 -> 500) on
     Adult alpha*, i.e. pool-exclusion bias for the V-LL selector is
     empirically negligible.

Writes one JSON per dataset to results/<ds>/uniqueness_check.json and a
top-level Markdown summary to results/uniqueness_check.md.

Per the plan in jair-results-script-5ae072.md:
    - Adult: alpha* = 1.5e-6 and the tightest feasible alpha = 1.0e-8
    - COMPAS: alpha* = 5.0e-7
    - Bank Marketing: alpha* = 5.0e-6

Pool-size stability is tested on Adult alpha* only.

Usage:
    python verify_uniqueness.py                 # all three datasets
    python verify_uniqueness.py --dataset adult # just one
    python verify_uniqueness.py --quick         # skip pool=500 stability
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure we can import fb_functions & pipeline regardless of cwd
PHD_DIR = Path(__file__).resolve().parent
if str(PHD_DIR) not in sys.path:
    sys.path.insert(0, str(PHD_DIR))

from fb_functions import DataPrep, GenVs, CalcMs, CalcBounds, AdjustEnfs  # noqa: E402
from pipeline import build_vll_scorer, solve_and_select  # noqa: E402


DATASET_CONFIG = {
    "adult": {
        "parquet": "data/adult_processed.parquet",
        "protected_vars": ["race_cat", "sex"],
        "alpha_star": 1.0e-7,
        "alpha_tightest": 1.0e-12,
    },
    "compas": {
        "parquet": "data/compas_processed.parquet",
        "protected_vars": ["race_cat", "sex"],
        "alpha_star": 2.0e-6,
        "alpha_tightest": 1.0e-12,
    },
    "bm": {
        "parquet": "data/bm_processed.parquet",
        "protected_vars": ["marital", "age_cat"],
        "alpha_star": 2.0e-6,
        "alpha_tightest": 1.0e-12,
    },
}

INTERVALS = 1000
HBASELINE = 100
A0 = 1
B0 = 1
X_VALUES = np.linspace(1 / (2 * INTERVALS), 1 - 1 / (2 * INTERVALS), INTERVALS)

CACHE_DIR_NAME = "cache"


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_dataset(ds: str, results_root: Path, force: bool = False):
    """Load raw data, run DataPrep/GenVs/CalcMs, cache dndf+vndf_base."""
    cfg = DATASET_CONFIG[ds]
    data_path = PHD_DIR / cfg["parquet"]
    ds_dir = _ensure_dir(results_root / ds)
    cache_dir = _ensure_dir(ds_dir / CACHE_DIR_NAME)
    dndf_cache = cache_dir / "dndf_base.pkl"
    vndf_cache = cache_dir / "vndf_base.pkl"

    if not force and dndf_cache.exists() and vndf_cache.exists():
        print(f"[{ds}] loading cached dndf/vndf_base ...")
        dndf = pd.read_pickle(dndf_cache)
        vndf_base = pd.read_pickle(vndf_cache)
    else:
        print(f"[{ds}] running DataPrep, GenVs, CalcMs (can take several minutes)")
        data = pd.read_parquet(data_path)
        dndf, _ = DataPrep(data, A0, B0, X_VALUES)
        vndf = GenVs(dndf, cfg["protected_vars"])
        vndf = CalcMs(data, dndf, vndf, HBASELINE, X_VALUES)
        vndf_base = vndf.copy()
        dndf.to_pickle(dndf_cache)
        vndf_base.to_pickle(vndf_cache)
    return dndf, vndf_base, cfg, ds_dir


def _run_alpha_check(
    alpha: float,
    dndf: pd.DataFrame,
    vndf_base: pd.DataFrame,
    score_fn,
    pool_solutions: int = 100,
    label: str | None = None,
) -> dict:
    """Solve at a single alpha, return uniqueness metrics."""
    if label is None:
        label = f"alpha={alpha:.2e}"
    print(f"  [{label}] CalcBounds + AdjustEnfs ...")
    vndf_a = CalcBounds(vndf_base.copy(), alpha)
    vndf_a = AdjustEnfs(vndf_a, dndf)

    print(f"  [{label}] solve_and_select (pool={pool_solutions}) ...")
    res = solve_and_select(
        vndf_a, vndf_base, dndf,
        pool_solutions=pool_solutions,
        score_fn=score_fn,
    )

    out = {
        "alpha": alpha,
        "pool_solutions_requested": pool_solutions,
        "feasible": bool(res.feasible),
        "n_constraints": int(res.n_constraints),
        "pool_size": int(res.pool_size),
        "mip_gap": res.mip_gap,
        "runtime_s": round(res.runtime, 2),
    }
    if res.feasible:
        out.update({
            "best_d_obj": float(res.d_obj[res.best_d_idx]),
            "best_v_ll": float(res.v_ll[res.best_v_idx]),
            "n_at_best_d": res.n_at_best_d,
            "n_at_best_v": res.n_at_best_v,
            "best_d_idx": res.best_d_idx,
            "best_v_idx": res.best_v_idx,
            "argmax_agree": bool(res.best_d_idx == res.best_v_idx),
            "pred_hamming_d_vs_v": int((res.pred_d != res.pred_v).sum()),
            "spearman_rho": res.spearman_rho,
            "spearman_p": res.spearman_p,
            "d_obj_range": [float(res.d_obj.min()), float(res.d_obj.max())],
            "v_ll_range": [float(res.v_ll.min()), float(res.v_ll.max())],
        })
    return out, res


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def verify_one_dataset(
    ds: str,
    results_root: Path,
    pool_solutions: int = 100,
    quick: bool = False,
    force: bool = False,
) -> dict:
    print(f"\n========== {ds.upper()} ==========")
    t_ds = time.time()
    dndf, vndf_base, cfg, ds_dir = _prepare_dataset(ds, results_root, force=force)
    print(f"[{ds}] d-nodes = {len(dndf):,}, v-nodes (base) = {len(vndf_base):,}")

    # Build V-LL scorer once per dataset
    score_fn, valid_idx, _ = build_vll_scorer(vndf_base, dndf)
    print(f"[{ds}] V-LL scorer built: {len(valid_idx):,} v-nodes with valid M")

    report = {
        "dataset": ds,
        "protected_vars": cfg["protected_vars"],
        "n_d_nodes": len(dndf),
        "n_v_nodes_base": len(vndf_base),
        "n_v_nodes_valid_M": int(len(valid_idx)),
        "alpha_star": cfg["alpha_star"],
        "alpha_tightest": cfg["alpha_tightest"],
        "pool_solutions": pool_solutions,
        "checks": {},
    }

    # ── Check 1: alpha* at the standard pool ──────────────────────────────
    c1, res_star = _run_alpha_check(
        cfg["alpha_star"], dndf, vndf_base, score_fn,
        pool_solutions=pool_solutions, label=f"{ds} alpha*",
    )
    report["checks"]["alpha_star"] = c1

    # ── Check 2: tightest feasible alpha (Adult only per plan; we include all
    #    for symmetry since the extra runtime is modest) ──────────────────
    if cfg["alpha_tightest"] != cfg["alpha_star"]:
        c2, _ = _run_alpha_check(
            cfg["alpha_tightest"], dndf, vndf_base, score_fn,
            pool_solutions=pool_solutions, label=f"{ds} tightest",
        )
        report["checks"]["alpha_tightest"] = c2

    # ── Check 3: Adult-only pool-size stability at alpha* ─────────────────
    if ds == "adult" and not quick:
        c3, res_big = _run_alpha_check(
            cfg["alpha_star"], dndf, vndf_base, score_fn,
            pool_solutions=500, label=f"{ds} alpha* pool=500",
        )
        # Compare best_v_ll and best prediction vector to pool=100 run
        if res_star.feasible and res_big.feasible:
            stable_vll = np.isclose(
                res_star.v_ll[res_star.best_v_idx],
                res_big.v_ll[res_big.best_v_idx],
                atol=1e-6,
            )
            pred_diff = int((res_star.pred_v != res_big.pred_v).sum())
            c3["stable_best_v_ll"] = bool(stable_vll)
            c3["pred_v_hamming_vs_pool100"] = pred_diff
            c3["stable_pred_v"] = bool(pred_diff == 0)
        report["checks"]["alpha_star_pool500"] = c3

    # ── Roll-up verdict ───────────────────────────────────────────────────
    verdict = _roll_up_verdict(report)
    report["verdict"] = verdict
    report["runtime_s"] = round(time.time() - t_ds, 2)

    # Persist
    out_path = ds_dir / "uniqueness_check.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[{ds}] wrote {out_path}")
    print(f"[{ds}] verdict: {verdict['status']}")
    for reason in verdict["reasons"]:
        print(f"         - {reason}")

    return report


def _roll_up_verdict(report: dict) -> dict:
    reasons = []
    for check_name, c in report["checks"].items():
        if not c.get("feasible", False):
            reasons.append(f"{check_name}: INFEASIBLE at alpha={c['alpha']:.1e}")
            continue
        if c.get("mip_gap") is not None and c["mip_gap"] > 1e-6:
            reasons.append(f"{check_name}: non-zero MIPGap = {c['mip_gap']:.3e}")
        if c.get("n_at_best_d", 0) > 1:
            reasons.append(f"{check_name}: N@Best_d = {c['n_at_best_d']} (not unique)")
        if c.get("n_at_best_v", 0) > 1:
            reasons.append(f"{check_name}: N@Best_v = {c['n_at_best_v']} (not unique)")
        if not c.get("argmax_agree", True):
            reasons.append(
                f"{check_name}: D-best != V-best (hamming "
                f"{c.get('pred_hamming_d_vs_v', '?')})"
            )
        if check_name == "alpha_star_pool500":
            if c.get("stable_best_v_ll") is False:
                reasons.append(f"{check_name}: V-LL unstable vs pool=100")
            if c.get("stable_pred_v") is False:
                reasons.append(f"{check_name}: pred_v differs vs pool=100")

    status = "PASS" if not reasons else "REVIEW"
    return {"status": status, "reasons": reasons or ["All checks passed."]}


def write_summary_md(all_reports: list[dict], results_root: Path) -> Path:
    lines = ["# Uniqueness verification summary\n"]
    for r in all_reports:
        lines.append(f"## {r['dataset']} — {r['verdict']['status']}\n")
        lines.append(f"- d-nodes: **{r['n_d_nodes']:,}**, v-nodes (base): **{r['n_v_nodes_base']:,}**, valid M: **{r['n_v_nodes_valid_M']:,}**")
        lines.append(f"- alpha* = **{r['alpha_star']:.1e}**, pool = **{r['pool_solutions']}**")
        lines.append("")

        lines.append("| Check | alpha | Feasible | MIPGap | Best D-obj | N@Best_d | Best V-LL | N@Best_v | D==V? | Hamming | Spearman rho |")
        lines.append("|-------|-------|----------|--------|-----------:|---------:|----------:|---------:|:-----:|--------:|-------------:|")
        for check_name, c in r["checks"].items():
            if not c.get("feasible", False):
                lines.append(f"| {check_name} | {c['alpha']:.1e} | No | - | - | - | - | - | - | - | - |")
                continue
            lines.append(
                "| {name} | {alpha:.1e} | Yes | {gap} | {d:.4f} | {nd} | {v:.2f} | {nv} | {agree} | {ham} | {rho} |".format(
                    name=check_name,
                    alpha=c["alpha"],
                    gap=(f"{c['mip_gap']:.1e}" if c['mip_gap'] is not None else "-"),
                    d=c["best_d_obj"],
                    nd=c["n_at_best_d"],
                    v=c["best_v_ll"],
                    nv=c["n_at_best_v"],
                    agree=("yes" if c["argmax_agree"] else "NO"),
                    ham=c["pred_hamming_d_vs_v"],
                    rho=(f"{c['spearman_rho']:.3f}" if c["spearman_rho"] is not None else "-"),
                )
            )
        if "alpha_star_pool500" in r["checks"]:
            c = r["checks"]["alpha_star_pool500"]
            if "stable_best_v_ll" in c:
                lines.append("")
                lines.append(
                    f"- Pool=500 stability: best V-LL stable = **{c['stable_best_v_ll']}**, "
                    f"pred_v differences vs pool=100 = **{c['pred_v_hamming_vs_pool100']}**"
                )
        lines.append("")
        lines.append("**Verdict**: " + r["verdict"]["status"])
        for reason in r["verdict"]["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    out = results_root / "uniqueness_check.md"
    out.write_text("\n".join(lines))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["adult", "compas", "bm", "all"], default="all"
    )
    parser.add_argument("--pool", type=int, default=100)
    parser.add_argument(
        "--quick", action="store_true",
        help="skip the pool=500 stability check",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="rebuild the CalcMs cache even if present",
    )
    parser.add_argument(
        "--results-root", type=str,
        default=str(PHD_DIR / "results"),
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    _ensure_dir(results_root)

    datasets = (
        ["adult", "compas", "bm"] if args.dataset == "all" else [args.dataset]
    )

    all_reports = []
    for ds in datasets:
        rep = verify_one_dataset(
            ds, results_root,
            pool_solutions=args.pool,
            quick=args.quick,
            force=args.force,
        )
        all_reports.append(rep)

    md_path = write_summary_md(all_reports, results_root)
    print(f"\nWrote {md_path}")

    # Non-zero exit if anything is REVIEW (so pipelines can short-circuit)
    failed = [r["dataset"] for r in all_reports if r["verdict"]["status"] != "PASS"]
    if failed:
        print(f"\n⚠  Datasets needing review: {failed}")
        sys.exit(1)
    print("\n✓ All verifications passed.")


if __name__ == "__main__":
    main()
