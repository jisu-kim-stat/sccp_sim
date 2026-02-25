#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# ----------------------------
# IMPORTANT: reuse your existing utilities
# ----------------------------
# Assumption: run_sccp_class_cluster.py defines these functions.
from run_sccp_class_cluster import (
    load_npz_probs,
    scores_softmax,
    eval_metrics,
    maybe_load_tail_set,
    tail_from_counts,
    build_sccp_index,
    predict_sets_lccp,
    predict_sets_cccp,
    predict_sets_sccp_class_cluster,
    kmeans_cluster_classes,
    class_quantile_embedding,
)

def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def stratified_split_indices(
    y: np.ndarray,
    sel_frac: float,
    cal_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split on labels y (1D int).
    Returns: idx_sel, idx_cal, idx_test (indices into original array)
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    n = y.size

    # sanity
    fr_sum = sel_frac + cal_frac + test_frac
    if not np.isclose(fr_sum, 1.0):
        raise ValueError(f"sel_frac+cal_frac+test_frac must sum to 1. Got {fr_sum}")

    idx_sel, idx_cal, idx_test = [], [], []
    classes = np.unique(y)

    for c in classes:
        idx_c = np.flatnonzero(y == c)
        rng.shuffle(idx_c)

        nc = idx_c.size
        n_sel = int(np.floor(sel_frac * nc))
        n_cal = int(np.floor(cal_frac * nc))
        # remainder to test
        n_test = nc - n_sel - n_cal

        if n_test < 0:
            raise RuntimeError("Negative n_test; check fractions.")

        idx_sel.append(idx_c[:n_sel])
        idx_cal.append(idx_c[n_sel:n_sel + n_cal])
        idx_test.append(idx_c[n_sel + n_cal:])

    idx_sel = np.concatenate(idx_sel) if len(idx_sel) else np.array([], dtype=int)
    idx_cal = np.concatenate(idx_cal) if len(idx_cal) else np.array([], dtype=int)
    idx_test = np.concatenate(idx_test) if len(idx_test) else np.array([], dtype=int)

    # final shuffle within each split (optional)
    rng.shuffle(idx_sel)
    rng.shuffle(idx_cal)
    rng.shuffle(idx_test)
    return idx_sel, idx_cal, idx_test


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)

    # embedding/clustering params (same as your SCCP script)
    ap.add_argument("--M", type=int, default=25)
    ap.add_argument("--n_clusters", type=int, default=20)
    ap.add_argument("--cluster_seed", type=int, default=1)

    # tau tuning
    ap.add_argument("--tau_grid", type=str, default="0,0.5,1,2,5,10,20,50,100,200")
    ap.add_argument("--tune_eps", type=float, default=0.01)

    # tail
    ap.add_argument("--tail_frac", type=float, default=0.2)

    # resplit control
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--resplit_seed0", type=int, default=1)
    ap.add_argument("--pool_mode", type=str, default="sel+cal+test",
                    choices=["sel+cal+test", "cal+test", "sel+cal"],
                    help="Which splits to merge as pool for resplitting. Recommended: sel+cal+test.")

    ap.add_argument("--sel_frac", type=float, default=0.1)
    ap.add_argument("--cal_frac", type=float, default=0.45)
    ap.add_argument("--test_frac", type=float, default=0.45)

    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--print_tau_table", action="store_true")

    args = ap.parse_args()

    # ----------------------------
    # Load original probs splits
    # ----------------------------
    splits = load_npz_probs(args.npz, K=args.K)
    P_sel, y_sel = splits["sel"]
    P_cal, y_cal = splits["cal"]
    P_test, y_test = splits["test"]

    # build pool
    if args.pool_mode == "sel+cal+test":
        P_pool = np.concatenate([P_sel, P_cal, P_test], axis=0)
        y_pool = np.concatenate([y_sel, y_cal, y_test], axis=0)
    elif args.pool_mode == "cal+test":
        P_pool = np.concatenate([P_cal, P_test], axis=0)
        y_pool = np.concatenate([y_cal, y_test], axis=0)
    else:  # sel+cal
        P_pool = np.concatenate([P_sel, P_cal], axis=0)
        y_pool = np.concatenate([y_sel, y_cal], axis=0)

    # scores (fixed, since P fixed)
    s_pool = scores_softmax(P_pool)

    # tail set (prefer NPZ info if exists; otherwise from pool counts)
    tail_set = maybe_load_tail_set(args.npz, K=args.K, tail_frac=args.tail_frac)
    if tail_set.size == 0:
        counts_pool = np.bincount(y_pool, minlength=args.K)
        tail_set = tail_from_counts(counts_pool, tail_frac=args.tail_frac)

    tau_grid = parse_csv_floats(args.tau_grid)
    if len(tau_grid) == 0:
        raise ValueError("tau_grid must be non-empty.")

    target = 1.0 - float(args.alpha)
    cov_min = target - float(args.tune_eps)

    # ----------------------------
    # Repeated resplits
    # ----------------------------
    rows = []

    for r in range(args.n_reps):
        rep_seed = args.resplit_seed0 + r

        idx_sel_r, idx_cal_r, idx_test_r = stratified_split_indices(
            y_pool,
            sel_frac=float(args.sel_frac),
            cal_frac=float(args.cal_frac),
            test_frac=float(args.test_frac),
            seed=rep_seed,
        )

        # create split arrays
        s_sel_r, y_sel_r = s_pool[idx_sel_r], y_pool[idx_sel_r]
        s_cal_r, y_cal_r = s_pool[idx_cal_r], y_pool[idx_cal_r]
        s_test_r, y_test_r = s_pool[idx_test_r], y_pool[idx_test_r]

        # ---- build class->cluster map c_y from CAL only
        n_cal = s_cal_r.shape[0]
        S_true_cal = s_cal_r[np.arange(n_cal), y_cal_r]     # (n_cal,)

        # quantile grid (M quantiles). use same convention as your main script
        q_grid = np.linspace(0.0, 1.0, int(args.M) + 2)[1:-1]  # (M,)

        Z, _ = class_quantile_embedding(S_true_cal, y_cal_r, K=args.K, q_grid=q_grid)  # (K,M)
        c_y = kmeans_cluster_classes(Z, n_clusters=int(args.n_clusters), seed=int(args.cluster_seed))  # (K,)

        # ---- build SCCP index
        idx = build_sccp_index(s_cal_r, y_cal_r, c_y, int(args.n_clusters))

        # baselines on TEST
        S_lccp = predict_sets_lccp(s_test_r, idx, alpha=args.alpha)
        met_lccp = eval_metrics(S_lccp, y_test_r, alpha=args.alpha, tail_set=tail_set)

        S_cccp = predict_sets_cccp(s_test_r, idx, alpha=args.alpha)
        met_cccp = eval_metrics(S_cccp, y_test_r, alpha=args.alpha, tail_set=tail_set)

        # tune tau on SEL (constrained)
        best_tau = None
        best = None
        best_sel_metrics = None
        any_feasible = False

        if args.print_tau_table:
            print("")
            print(f"[rep {r+1}/{args.n_reps}] seed={rep_seed}")
            print("tau | sel_marg_cov | sel_cov_tail | sel_worst | sel_avg_size | sel_tail_size | feasible")
            print("-" * 86)

        for tau in tau_grid:
            S_sel = predict_sets_sccp_class_cluster(s_sel_r, idx, tau=float(tau), alpha=args.alpha)
            met_sel = eval_metrics(S_sel, y_sel_r, alpha=args.alpha, tail_set=tail_set)

            marg = float(met_sel["marginal_cov"])
            feasible = (marg >= cov_min)

            if args.print_tau_table:
                print(f"{tau:>4g} | {marg:>11.4f} | {met_sel['cov_tail']:>11.4f} | "
                      f"{met_sel['worst_class_cov']:>9.4f} | {met_sel['avg_size']:>12.2f} | "
                      f"{met_sel['size_tail']:>13.2f} | {str(feasible)}")

            if feasible:
                any_feasible = True
                # NOTE: replace with your "avg coverage objective" if you already changed it
                obj = (
                    abs(met_sel["avg_class_cov"] - target),
                    met_sel["avg_size"],
                    -met_sel["worst_class_cov"],
                )
                if best is None or obj < best:
                    best = obj
                    best_tau = float(tau)
                    best_sel_metrics = met_sel

        # fallback if none feasible
        if not any_feasible:
            for tau in tau_grid:
                S_sel = predict_sets_sccp_class_cluster(s_sel_r, idx, tau=float(tau), alpha=args.alpha)
                met_sel = eval_metrics(S_sel, y_sel_r, alpha=args.alpha, tail_set=tail_set)
                obj = (
                    abs(met_sel["avg_class_cov"] - target),
                    met_sel["avg_size"],
                    -met_sel["worst_class_cov"],
                )
                if best is None or obj < best:
                    best = obj
                    best_tau = float(tau)
                    best_sel_metrics = met_sel

        # proposed on TEST with tuned tau
        S_sccp = predict_sets_sccp_class_cluster(s_test_r, idx, tau=float(best_tau), alpha=args.alpha)
        met_sccp = eval_metrics(S_sccp, y_test_r, alpha=args.alpha, tail_set=tail_set)

        rows.append({
            "rep": r,
            "rep_seed": rep_seed,
            "best_tau": float(best_tau),

            # LCCP
            "lccp_marginal_cov": met_lccp["marginal_cov"],
            "lccp_avg_size": met_lccp["avg_size"],
            "lccp_avg_class_cov": met_lccp["avg_class_cov"],
            "lccp_worst_class_cov": met_lccp["worst_class_cov"],
            "lccp_maxgap": met_lccp["maxgap"],
            "lccp_cov_tail": met_lccp.get("cov_tail", np.nan),
            "lccp_size_tail": met_lccp.get("size_tail", np.nan),

            # CCCP
            "cccp_marginal_cov": met_cccp["marginal_cov"],
            "cccp_avg_size": met_cccp["avg_size"],
            "cccp_avg_class_cov": met_cccp["avg_class_cov"],
            "cccp_worst_class_cov": met_cccp["worst_class_cov"],
            "cccp_maxgap": met_cccp["maxgap"],
            "cccp_cov_tail": met_cccp.get("cov_tail", np.nan),
            "cccp_size_tail": met_cccp.get("size_tail", np.nan),

            # SCCP
            "sccp_marginal_cov": met_sccp["marginal_cov"],
            "sccp_avg_size": met_sccp["avg_size"],
            "sccp_avg_class_cov": met_sccp["avg_class_cov"],
            "sccp_worst_class_cov": met_sccp["worst_class_cov"],
            "sccp_maxgap": met_sccp["maxgap"],
            "sccp_cov_tail": met_sccp.get("cov_tail", np.nan),
            "sccp_size_tail": met_sccp.get("size_tail", np.nan),
        })

    # ----------------------------
    # Save outputs
    # ----------------------------
    out = {
        "npz": args.npz,
        "K_classes": int(args.K),
        "alpha": float(args.alpha),
        "tail_frac": float(args.tail_frac),
        "tail_set": tail_set.tolist(),
        "pool_mode": args.pool_mode,
        "resplit": {
            "n_reps": int(args.n_reps),
            "resplit_seed0": int(args.resplit_seed0),
            "sel_frac": float(args.sel_frac),
            "cal_frac": float(args.cal_frac),
            "test_frac": float(args.test_frac),
        },
        "clustering": {
            "method": "kmeans",
            "M": int(args.M),
            "n_clusters": int(args.n_clusters),
            "cluster_seed": int(args.cluster_seed),
        },
        "tuning": {
            "tau_grid": [float(t) for t in tau_grid],
            "constraint": {
                "target": float(target),
                "eps": float(args.tune_eps),
                "min_marginal_cov": float(cov_min),
            },
            "objective": "abs(avg_class_cov - target), then avg_size, then -worst_class_cov",
        },
        "rows": rows,
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}")

    if args.out_csv:
        import csv
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()