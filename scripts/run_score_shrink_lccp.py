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
    return isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == K and np.isfinite(a).all()

def _is_label_vector(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 1 and np.issubdtype(a.dtype, np.integer)

def _normalize_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
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
        keys = [k for k in d.keys() if tag in k.lower() and ("p_" in k.lower() or "prob" in k.lower() or "probs" in k.lower())]
        # Prefer exact matches like p_sel
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
# Ranking utilities: u = (1 + #{cal >= value}) / (n + 1)
# Implemented via sorting + binary search.
# ============================================================

@dataclass
class RankIndex:
    # For each class y:
    #   - sorted scores for class-conditional (only cal points with Y=y)
    #   - sorted scores for global (all cal points, but score column y)
    s_class_sorted: List[np.ndarray]  # length K, each shape (n_y,)
    s_global_sorted: List[np.ndarray] # length K, each shape (n_cal,)
    z_class_sorted: Optional[List[np.ndarray]] = None  # after we compute z_i for cal (true label only), per class y
    n_y: Optional[np.ndarray] = None
    n_cal: int = 0

def _u_from_sorted_ge(sorted_arr: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    sorted_arr: 1D ascending array of calibration scores.
    Return u(v) = (1 + #{sorted_arr >= v}) / (n + 1)
    For vector v: use searchsorted to find leftmost index of v in ascending order.
    ge_count = n - idx_left
    """
    n = sorted_arr.size
    if n == 0:
        # Shouldn't happen in our setting (each label exists), but keep safe.
        return np.ones_like(v, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    idx_left = np.searchsorted(sorted_arr, v, side="left")  # first index where sorted_arr >= v
    ge = (n - idx_left).astype(np.float64)
    return (1.0 + ge) / (n + 1.0)

def build_rank_index(scores_cal: np.ndarray, y_cal: np.ndarray) -> RankIndex:
    """
    scores_cal: (n_cal, K) nonconformity matrix on calibration set
    y_cal: (n_cal,)
    """
    scores_cal = np.asarray(scores_cal, dtype=np.float64)
    y_cal = np.asarray(y_cal, dtype=int)
    n_cal, K = scores_cal.shape

    s_global_sorted: List[np.ndarray] = []
    for y in range(K):
        s_global_sorted.append(np.sort(scores_cal[:, y], axis=0))

    s_class_sorted: List[np.ndarray] = []
    n_y = np.zeros(K, dtype=int)
    for y in range(K):
        m = (y_cal == y)
        n_y[y] = int(m.sum())
        s_class_sorted.append(np.sort(scores_cal[m, y], axis=0))

    return RankIndex(
        s_class_sorted=s_class_sorted,
        s_global_sorted=s_global_sorted,
        z_class_sorted=None,
        n_y=n_y,
        n_cal=n_cal,
    )


# ============================================================
# Proposed method: score-level shrinkage
# z(x,y) = -log u_c(x,y) + beta_y * (-log u_g(x,y))
# beta_y = tau / (tau + n_y)
# Final p-value computed by classwise ranking of z among cal points with Y=y.
# ============================================================

def beta_from_counts(n_y: np.ndarray, tau: float) -> np.ndarray:
    n_y = np.asarray(n_y, dtype=np.float64)
    tau = float(tau)
    return tau / (tau + n_y)

def compute_z_for_matrix(
    scores_X: np.ndarray,  # (n, K)
    idx: RankIndex,
    beta: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    For each (i,y), compute u_c and u_g using precomputed sorted arrays,
    then compute z(i,y).
    """
    scores_X = np.asarray(scores_X, dtype=np.float64)
    n, K = scores_X.shape
    out = np.empty((n, K), dtype=np.float64)

    for y in range(K):
        uc = _u_from_sorted_ge(idx.s_class_sorted[y], scores_X[:, y])
        ug = _u_from_sorted_ge(idx.s_global_sorted[y], scores_X[:, y])
        uc = np.clip(uc, eps, 1.0)
        ug = np.clip(ug, eps, 1.0)
        out[:, y] = -np.log(uc) + float(beta[y]) * (-np.log(ug))
    return out

def fit_z_calibration_index(
    scores_cal: np.ndarray, y_cal: np.ndarray, tau: float
) -> Tuple[RankIndex, np.ndarray]:
    """
    1) Build rank index for u_c and u_g from calibration scores.
    2) Compute z(x,y) for all cal points and all y.
    3) For each class y, collect z_i,true among cal points with Y=y and sort (for final p-values).
    """
    idx = build_rank_index(scores_cal, y_cal)
    beta = beta_from_counts(idx.n_y, tau=tau)

    z_cal_all = compute_z_for_matrix(scores_cal, idx, beta=beta)  # (n_cal, K)
    z_true = z_cal_all[np.arange(idx.n_cal), y_cal]              # (n_cal,)

    z_class_sorted: List[np.ndarray] = []
    K = scores_cal.shape[1]
    for y in range(K):
        m = (y_cal == y)
        z_class_sorted.append(np.sort(z_true[m], axis=0))

    idx.z_class_sorted = z_class_sorted
    return idx, beta


# ============================================================
# Baseline LCCP: prediction set via u_c(x,y) > alpha
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
# Proposed: ScoreShrink-LCCP: compute z -> final classwise p-values on z
# ============================================================

def predict_sets_score_shrink_lccp(
    scores_X: np.ndarray,
    idx_fit: RankIndex,
    beta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if idx_fit.z_class_sorted is None:
        raise RuntimeError("idx_fit has no z_class_sorted. Did you call fit_z_calibration_index()?")

    scores_X = np.asarray(scores_X, dtype=np.float64)
    n, K = scores_X.shape
    z = compute_z_for_matrix(scores_X, idx_fit, beta=beta)  # (n,K)

    S = np.zeros((n, K), dtype=bool)
    for y in range(K):
        # final p-value via classwise ranking of z among cal points with Y=y:
        # u_hat = (1 + #{z_cal_true(y) >= z(x,y)}) / (n_y + 1)
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
# Tail set helper (optional from counts_pool in npz, else from y_cal+y_sel+y_test)
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
        ts = np.asarray(raw["tail_set"], dtype=int)
        return ts
    if "counts_pool" in raw.files:
        return tail_from_counts(raw["counts_pool"], tail_frac=tail_frac)
    # fallback: cannot infer here
    return np.array([], dtype=int)


# ============================================================
# Main: tune tau on sel, fit on cal, eval on test
# ============================================================

def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)

    ap.add_argument("--tau_grid", type=str, default="0,0.5,1,2,5,10,20,50,100,200",
                    help="Comma-separated tau candidates. tau=0 should reproduce baseline-ish behavior.")
    ap.add_argument("--tail_frac", type=float, default=0.2)

    ap.add_argument("--out_json", type=str, default="", help="Optional: save results to json")

    args = ap.parse_args()

    splits = load_npz_probs(args.npz, K=args.K)
    P_sel, y_sel = splits["sel"]
    P_cal, y_cal = splits["cal"]
    P_test, y_test = splits["test"]

    # scores
    s_sel = scores_softmax(P_sel)
    s_cal = scores_softmax(P_cal)
    s_test = scores_softmax(P_test)

    # tail set (optional)
    tail_set = maybe_load_tail_set(args.npz, K=args.K, tail_frac=args.tail_frac)
    if tail_set.size == 0:
        # fallback: define tail based on combined label counts (sel+cal+test)
        y_all = np.concatenate([y_sel, y_cal, y_test], axis=0)
        counts = np.bincount(y_all, minlength=args.K)
        tail_set = tail_from_counts(counts, tail_frac=args.tail_frac)

    # Baseline index from cal (for u_c)
    idx_base = build_rank_index(s_cal, y_cal)

    # ---- Baseline evaluation (LCCP)
    S_base_test = predict_sets_lccp(s_test, idx_base, alpha=args.alpha)
    met_base = eval_metrics(S_base_test, y_test, alpha=args.alpha, tail_set=tail_set)

    # ---- Tune tau on selection split
    tau_grid = parse_csv_floats(args.tau_grid)
    if len(tau_grid) == 0:
        raise ValueError("tau_grid must be non-empty.")

    best = None
    best_tau = None
    best_sel_metrics = None

    print(f"[file] {args.npz}")
    print(f"[K] {args.K}  [alpha] {args.alpha}")
    print(f"[tail] frac={args.tail_frac}  m={len(tail_set)}")
    print("")
    print("=== Baseline: LCCP (classwise p-values on softmax score) ===")
    print(json.dumps(met_base, indent=2))
    print("")

    for tau in tau_grid:
        # Fit z-index using CAL only (important: no sel leakage into conformalization)
        idx_fit, beta = fit_z_calibration_index(s_cal, y_cal, tau=float(tau))

        # Evaluate on SEL for tuning objective
        S_sel = predict_sets_score_shrink_lccp(s_sel, idx_fit, beta=beta, alpha=args.alpha)
        met_sel = eval_metrics(S_sel, y_sel, alpha=args.alpha, tail_set=tail_set)

        # Objective: prioritize tail coverage and worst-class coverage; break ties by avg_size
        # (You can change this easily.)
        obj = (
            -(met_sel.get("cov_tail", float("nan"))),          # maximize tail cov
            -(met_sel.get("worst_class_cov", float("nan"))),   # maximize worst-class cov
            met_sel.get("avg_size", float("inf"))              # minimize size
        )

        if best is None or obj < best:
            best = obj
            best_tau = float(tau)
            best_sel_metrics = met_sel

    print("=== Tuning on SEL ===")
    print(f"[best_tau] {best_tau}")
    print("[sel_metrics_best_tau]")
    print(json.dumps(best_sel_metrics, indent=2))
    print("")

    # ---- Refit on CAL with best tau, evaluate on TEST
    idx_fit, beta = fit_z_calibration_index(s_cal, y_cal, tau=float(best_tau))
    S_prop_test = predict_sets_score_shrink_lccp(s_test, idx_fit, beta=beta, alpha=args.alpha)
    met_prop = eval_metrics(S_prop_test, y_test, alpha=args.alpha, tail_set=tail_set)

    print("=== Proposed: ScoreShrink-LCCP on TEST ===")
    print(json.dumps(met_prop, indent=2))

    # Optional save
    if args.out_json:
        out = {
            "npz": args.npz,
            "K": int(args.K),
            "alpha": float(args.alpha),
            "tail_frac": float(args.tail_frac),
            "tail_set": tail_set.tolist(),
            "baseline_LCCP": met_base,
            "tuning": {
                "tau_grid": [float(t) for t in tau_grid],
                "best_tau": float(best_tau),
                "sel_metrics_best_tau": best_sel_metrics,
            },
            "proposed_ScoreShrink_LCCP": met_prop,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
