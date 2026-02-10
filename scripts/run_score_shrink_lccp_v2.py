#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# NPZ loading (expects p_sel,y_sel,p_cal,y_cal,p_test,y_test)
# ============================================================

def _is_prob_matrix(a: np.ndarray, K: int) -> bool:
    return (
        isinstance(a, np.ndarray)
        and a.ndim == 2
        and a.shape[1] == K
        and np.isfinite(a).all()
    )

def _is_label_vector(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 1 and np.issubdtype(a.dtype, np.integer)

def _normalize_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    s = P.sum(axis=1, keepdims=True)
    s = np.where(s <= eps, 1.0, s)
    Pn = P / s
    Pn = np.clip(Pn, eps, 1.0)
    Pn = Pn / Pn.sum(axis=1, keepdims=True)
    return Pn

def load_npz_probs(path: str, K: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    raw = np.load(path, allow_pickle=True)
    d = {k: raw[k] for k in raw.files}

    def pick_prob(tag: str) -> np.ndarray:
        keys = [
            k for k in d.keys()
            if tag in k.lower() and ("p_" in k.lower() or "prob" in k.lower() or "probs" in k.lower())
        ]
        for k in [f"p_{tag}", f"prob_{tag}", f"probs_{tag}", f"P_{tag}"]:
            if k in d and _is_prob_matrix(d[k], K):
                return d[k]
        for k in keys:
            if _is_prob_matrix(d[k], K):
                return d[k]
        raise RuntimeError(f"Could not find prob matrix for '{tag}' in NPZ. keys={list(d.keys())}")

    def pick_lab(tag: str) -> np.ndarray:
        for k in [f"y_{tag}", f"label_{tag}", f"labels_{tag}", f"Y_{tag}"]:
            if k in d and _is_label_vector(d[k]):
                return d[k]
        keys = [k for k in d.keys() if tag in k.lower() and ("y_" in k.lower() or "label" in k.lower())]
        for k in keys:
            if _is_label_vector(d[k]):
                return d[k]
        raise RuntimeError(f"Could not find label vector for '{tag}' in NPZ. keys={list(d.keys())}")

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tag in ["sel", "cal", "test"]:
        P = pick_prob(tag).astype(np.float64)
        y = pick_lab(tag).astype(int)
        if P.shape[0] != y.shape[0]:
            raise RuntimeError(f"Shape mismatch for '{tag}': P {P.shape} vs y {y.shape}")
        out[tag] = (_normalize_rows(P), y)

    return out


# ============================================================
# Score: softmax nonconformity s(x,y) = 1 - p_y(x)
# ============================================================

def scores_softmax(P: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(P, dtype=np.float64)


# ============================================================
# Global difficulty (x-only): entropy(P(x))
# g(x) large => "hard" example (more uniform probs)
# ============================================================

def entropy_from_probs(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    P = np.clip(P, eps, 1.0)
    return -np.sum(P * np.log(P), axis=1)  # (n,)


# ============================================================
# Ranking utilities: u = (1 + #{cal >= value}) / (n + 1)
# ============================================================

@dataclass
class RankIndex:
    # For each class y:
    #   - s_class_sorted[y] : sorted scores s_i(y) for cal points with Y=y
    #   - g_sorted          : sorted global difficulty g_i over ALL cal points
    s_class_sorted: List[np.ndarray]              # length K, each shape (n_y,)
    g_sorted: np.ndarray                          # shape (n_cal,)
    z_class_sorted: Optional[List[np.ndarray]] = None  # sorted z_true per class y
    n_y: Optional[np.ndarray] = None
    n_cal: int = 0

def _u_from_sorted_ge(sorted_arr: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    sorted_arr: 1D ascending array (calibration values)
    u(v) = (1 + #{sorted_arr >= v}) / (n + 1)
    """
    n = int(sorted_arr.size)
    if n == 0:
        return np.ones_like(v, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    idx_left = np.searchsorted(sorted_arr, v, side="left")  # first index where sorted_arr >= v
    ge = (n - idx_left).astype(np.float64)
    return (1.0 + ge) / (n + 1.0)

def build_rank_index(scores_cal: np.ndarray, y_cal: np.ndarray, g_cal: np.ndarray) -> RankIndex:
    """
    scores_cal: (n_cal, K)
    y_cal: (n_cal,)
    g_cal: (n_cal,)
    """
    scores_cal = np.asarray(scores_cal, dtype=np.float64)
    y_cal = np.asarray(y_cal, dtype=int)
    g_cal = np.asarray(g_cal, dtype=np.float64)

    n_cal, K = scores_cal.shape

    s_class_sorted: List[np.ndarray] = []
    n_y = np.zeros(K, dtype=int)
    for y in range(K):
        m = (y_cal == y)
        n_y[y] = int(m.sum())
        s_class_sorted.append(np.sort(scores_cal[m, y], axis=0))

    return RankIndex(
        s_class_sorted=s_class_sorted,
        g_sorted=np.sort(g_cal, axis=0),
        z_class_sorted=None,
        n_y=n_y,
        n_cal=n_cal,
    )


# ============================================================
# Proposed method (D2-style): score-level shrinkage using x-only global info
#
# uc(x,y) = (1 + #{cal, Y=y : s_cal(y) >= s(x,y)})/(n_y+1)
# ug(x)   = (1 + #{cal : g_cal >= g(x)})/(n_cal+1)
#
# z(x,y) = -log uc(x,y)  - beta_y * (-log ug(x))
# beta_y = tau/(tau+n_y)
#
# Final p-value: classwise ranking of z among z_true for cal points with Y=y:
# uhat(x,y) = (1 + #{cal, Y=y : z_true >= z(x,y)})/(n_y+1)
# include y if uhat(x,y) > alpha
# ============================================================

def beta_from_counts(n_y: np.ndarray, tau: float) -> np.ndarray:
    n_y = np.asarray(n_y, dtype=np.float64)
    tau = float(tau)
    return tau / (tau + n_y)

def compute_z_for_matrix(
    scores_X: np.ndarray,     # (n, K)
    g_X: np.ndarray,          # (n,)
    idx: RankIndex,
    beta: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    scores_X = np.asarray(scores_X, dtype=np.float64)
    g_X = np.asarray(g_X, dtype=np.float64)
    n, K = scores_X.shape

    # x-only global ranking
    ug = _u_from_sorted_ge(idx.g_sorted, g_X)
    ug = np.clip(ug, eps, 1.0)
    glog = -np.log(ug)  # (n,)

    out = np.empty((n, K), dtype=np.float64)
    for y in range(K):
        uc = _u_from_sorted_ge(idx.s_class_sorted[y], scores_X[:, y])
        uc = np.clip(uc, eps, 1.0)
        # D2: "-" (harder x => smaller ug => larger glog => z decreases => uhat increases => sets shrink)
        out[:, y] = -np.log(uc) - float(beta[y]) * glog

    return out

def fit_z_calibration_index(
    scores_cal: np.ndarray, y_cal: np.ndarray, g_cal: np.ndarray, tau: float
) -> Tuple[RankIndex, np.ndarray]:
    idx = build_rank_index(scores_cal, y_cal, g_cal=g_cal)
    beta = beta_from_counts(idx.n_y, tau=tau)

    z_cal_all = compute_z_for_matrix(scores_cal, g_cal, idx, beta=beta)  # (n_cal,K)
    z_true = z_cal_all[np.arange(idx.n_cal), y_cal]                      # (n_cal,)

    z_class_sorted: List[np.ndarray] = []
    K = scores_cal.shape[1]
    for y in range(K):
        m = (y_cal == y)
        z_class_sorted.append(np.sort(z_true[m], axis=0))

    idx.z_class_sorted = z_class_sorted
    return idx, beta


# ============================================================
# Baseline LCCP: prediction set via uc(x,y) > alpha
# ============================================================

def predict_sets_lccp(scores_X: np.ndarray, idx: RankIndex, alpha: float) -> np.ndarray:
    scores_X = np.asarray(scores_X, dtype=np.float64)
    n, K = scores_X.shape
    S = np.zeros((n, K), dtype=bool)
    for y in range(K):
        uc = _u_from_sorted_ge(idx.s_class_sorted[y], scores_X[:, y])
        S[:, y] = (uc > float(alpha))
    return S


# ============================================================
# Proposed: ScoreShrink-LCCP (D2 global info)
# ============================================================

def predict_sets_score_shrink_lccp(
    scores_X: np.ndarray,
    g_X: np.ndarray,
    idx_fit: RankIndex,
    beta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if idx_fit.z_class_sorted is None:
        raise RuntimeError("idx_fit has no z_class_sorted. Did you call fit_z_calibration_index()?")

    z = compute_z_for_matrix(scores_X, g_X, idx_fit, beta=beta)  # (n,K)

    n, K = z.shape
    S = np.zeros((n, K), dtype=bool)
    for y in range(K):
        uhat = _u_from_sorted_ge(idx_fit.z_class_sorted[y], z[:, y])
        S[:, y] = (uhat > float(alpha))
    return S


# ============================================================
# Evaluation
# ============================================================

def eval_metrics(S: np.ndarray, y_true: np.ndarray, alpha: float, tail_set: Optional[np.ndarray] = None) -> Dict[str, float]:
    S = np.asarray(S, dtype=bool)
    y_true = np.asarray(y_true, dtype=int)
    n, K = S.shape

    hit = S[np.arange(n), y_true].astype(float)
    sizes = S.sum(axis=1).astype(float)

    cov_k = np.full(K, np.nan, dtype=float)
    n_k = np.zeros(K, dtype=int)
    for k in range(K):
        m = (y_true == k)
        n_k[k] = int(m.sum())
        if n_k[k] > 0:
            cov_k[k] = float(np.mean(hit[m]))

    out = {
        "marginal_cov": float(np.mean(hit)),
        "avg_size": float(np.mean(sizes)),
        "avg_class_cov": float(np.nanmean(cov_k)),
        "worst_class_cov": float(np.nanmin(cov_k)),
        "std_class_cov": float(np.nanstd(cov_k)),
        "covgap": float(np.nanmean(np.abs(cov_k - (1.0 - float(alpha))))),
        "maxgap": float(np.nanmax(np.abs(cov_k - (1.0 - float(alpha))))),
    }

    if tail_set is not None:
        tail_set = np.asarray(tail_set, dtype=int)
        is_tail = np.isin(y_true, tail_set)
        is_head = ~is_tail
        out.update({
            "n_tail": int(is_tail.sum()),
            "n_head": int(is_head.sum()),
            "cov_tail": float(np.mean(hit[is_tail])) if is_tail.any() else float("nan"),
            "cov_head": float(np.mean(hit[is_head])) if is_head.any() else float("nan"),
            "size_tail": float(np.mean(sizes[is_tail])) if is_tail.any() else float("nan"),
            "size_head": float(np.mean(sizes[is_head])) if is_head.any() else float("nan"),
        })
    return out


# ============================================================
# Tail set helper
# ============================================================

def tail_from_counts(counts: np.ndarray, tail_frac: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    K = counts.shape[0]
    m = int(np.ceil(float(tail_frac) * K))
    m = max(0, min(m, K))
    order = np.argsort(counts)  # ascending
    return order[:m].astype(int)

def maybe_load_tail_set(npz_path: str, K: int, tail_frac: float) -> np.ndarray:
    raw = np.load(npz_path, allow_pickle=True)
    if "tail_set" in raw.files:
        return np.asarray(raw["tail_set"], dtype=int)
    if "counts_pool" in raw.files:
        return tail_from_counts(raw["counts_pool"], tail_frac=tail_frac)
    return np.array([], dtype=int)


# ============================================================
# Main: tune tau on SEL, fit on CAL, eval on TEST
# ============================================================

def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)

    ap.add_argument("--tau_grid", type=str, default="0,0.5,1,2,5,10,20,50,100,200")
    ap.add_argument("--tail_frac", type=float, default=0.2)

    ap.add_argument("--tune_eps", type=float, default=0.01,
                    help="Constraint slack eps for marginal coverage: require marginal_cov >= 1-alpha-eps")
    ap.add_argument("--print_tau_table", action="store_true")

    ap.add_argument("--out_json", type=str, default="", help="Optional: save results to json")

    args = ap.parse_args()

    splits = load_npz_probs(args.npz, K=args.K)
    P_sel, y_sel = splits["sel"]
    P_cal, y_cal = splits["cal"]
    P_test, y_test = splits["test"]

    # nonconformity scores
    s_sel = scores_softmax(P_sel)
    s_cal = scores_softmax(P_cal)
    s_test = scores_softmax(P_test)

    # x-only global difficulty
    g_sel = entropy_from_probs(P_sel)
    g_cal = entropy_from_probs(P_cal)
    g_test = entropy_from_probs(P_test)

    # tail set
    tail_set = maybe_load_tail_set(args.npz, K=args.K, tail_frac=args.tail_frac)
    if tail_set.size == 0:
        y_all = np.concatenate([y_sel, y_cal, y_test], axis=0)
        counts = np.bincount(y_all, minlength=args.K)
        tail_set = tail_from_counts(counts, tail_frac=args.tail_frac)

    # Baseline index from CAL (only classwise score ranks)
    idx_base = build_rank_index(s_cal, y_cal, g_cal=g_cal)

    # ---- Baseline evaluation (LCCP)
    S_base_test = predict_sets_lccp(s_test, idx_base, alpha=args.alpha)
    met_base = eval_metrics(S_base_test, y_test, alpha=args.alpha, tail_set=tail_set)

    # ---- Tune tau on SEL (constrained)
    tau_grid = parse_csv_floats(args.tau_grid)
    if len(tau_grid) == 0:
        raise ValueError("tau_grid must be non-empty.")

    target = 1.0 - float(args.alpha)
    eps = float(args.tune_eps)
    cov_min = target - eps

    best = None
    best_tau = None
    best_sel_metrics = None
    any_feasible = False

    print(f"[file] {args.npz}")
    print(f"[K] {args.K}  [alpha] {args.alpha}")
    print(f"[tail] frac={args.tail_frac}  m={len(tail_set)}")
    print("")
    print("=== Baseline: LCCP (classwise p-values on softmax score) ===")
    print(json.dumps(met_base, indent=2))
    print("")

    if args.print_tau_table:
        print("tau | sel_marg_cov | sel_cov_tail | sel_worst | sel_avg_size | sel_tail_size | feasible")
        print("-" * 86)

    for tau in tau_grid:
        # Fit z-index using CAL only (no sel leakage into conformalization)
        idx_fit, beta = fit_z_calibration_index(s_cal, y_cal, g_cal=g_cal, tau=float(tau))

        # Evaluate on SEL for tuning
        S_sel = predict_sets_score_shrink_lccp(s_sel, g_sel, idx_fit, beta=beta, alpha=args.alpha)
        met_sel = eval_metrics(S_sel, y_sel, alpha=args.alpha, tail_set=tail_set)

        marg = float(met_sel["marginal_cov"])
        feasible = (marg >= cov_min)

        if args.print_tau_table:
            print(f"{tau:>4g} | {marg:>11.4f} | {met_sel['cov_tail']:>11.4f} | "
                  f"{met_sel['worst_class_cov']:>9.4f} | {met_sel['avg_size']:>12.2f} | "
                  f"{met_sel['size_tail']:>13.2f} | {str(feasible)}")

        if feasible:
            any_feasible = True
            # Primary: minimize tail size; then minimize avg size; then maximize worst-class coverage
            obj = (
                met_sel.get("size_tail", float("inf")),
                met_sel.get("avg_size", float("inf")),
                -met_sel.get("worst_class_cov", float("nan")),
            )
            if best is None or obj < best:
                best = obj
                best_tau = float(tau)
                best_sel_metrics = met_sel

    print("=== Tuning on SEL (constrained) ===")
    print(f"[constraint] marginal_cov >= {cov_min:.4f}  (target={target:.4f}, eps={eps:.4f})")

    # If nothing feasible, fallback: maximize marginal_cov; tie-break by tail size then avg size.
    if not any_feasible:
        best = None
        for tau in tau_grid:
            idx_fit, beta = fit_z_calibration_index(s_cal, y_cal, g_cal=g_cal, tau=float(tau))
            S_sel = predict_sets_score_shrink_lccp(s_sel, g_sel, idx_fit, beta=beta, alpha=args.alpha)
            met_sel = eval_metrics(S_sel, y_sel, alpha=args.alpha, tail_set=tail_set)

            obj = (
                -met_sel.get("marginal_cov", float("-inf")),
                met_sel.get("size_tail", float("inf")),
                met_sel.get("avg_size", float("inf")),
            )
            if best is None or obj < best:
                best = obj
                best_tau = float(tau)
                best_sel_metrics = met_sel

    print(f"[best_tau] {best_tau}")
    print("[sel_metrics_best_tau]")
    print(json.dumps(best_sel_metrics, indent=2))
    print("")

    # ---- Refit on CAL with best tau, evaluate on TEST
    idx_fit, beta = fit_z_calibration_index(s_cal, y_cal, g_cal=g_cal, tau=float(best_tau))
    S_prop_test = predict_sets_score_shrink_lccp(s_test, g_test, idx_fit, beta=beta, alpha=args.alpha)
    met_prop = eval_metrics(S_prop_test, y_test, alpha=args.alpha, tail_set=tail_set)

    print("=== Proposed: ScoreShrink-LCCP (D2, entropy global) on TEST ===")
    print(json.dumps(met_prop, indent=2))

    # Optional save
    if args.out_json:
        out = {
            "npz": args.npz,
            "K": int(args.K),
            "alpha": float(args.alpha),
            "tail_frac": float(args.tail_frac),
            "tail_set": tail_set.tolist(),
            "global_difficulty": "entropy(P(x))",
            "baseline_LCCP": met_base,
            "tuning": {
                "tau_grid": [float(t) for t in tau_grid],
                "best_tau": float(best_tau),
                "constraint": {
                    "target": float(target),
                    "eps": float(eps),
                    "min_marginal_cov": float(cov_min),
                },
                "sel_metrics_best_tau": best_sel_metrics,
            },
            "proposed_ScoreShrink_LCCP_D2": met_prop,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
