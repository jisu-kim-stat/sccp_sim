#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# optional but recommended for k-means
from sklearn.cluster import KMeans


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
# Ranking utility: u = (1 + #{cal >= value}) / (n + 1)
# This is the standard conformal-style (smoothed) survival rank.
# ============================================================

def _u_from_sorted_ge(sorted_arr: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    sorted_arr: 1D ascending array (calibration values)
    u(v) = (1 + #{sorted_arr >= v}) / (n + 1)
    """
    n = int(sorted_arr.size)
    v = np.asarray(v, dtype=np.float64)

    if n == 0:
        # if no calibration points, return 1 (maximal p-value) => always included
        return np.ones_like(v, dtype=np.float64)

    idx_left = np.searchsorted(sorted_arr, v, side="left")  # first index where sorted_arr >= v
    ge = (n - idx_left).astype(np.float64)
    return (1.0 + ge) / (n + 1.0)


# ============================================================
# Quantile embedding + k-means clustering
# ============================================================

def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def make_quantile_grid(M: int) -> np.ndarray:
    # avoid endpoints for stability
    # q_j = j/(M+1), j=1..M
    if M <= 0:
        raise ValueError("M must be positive.")
    j = np.arange(1, M + 1, dtype=np.float64)
    return j / (M + 1.0)


def class_quantile_embedding(S_true: np.ndarray, y_cal: np.ndarray, K: int, q_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    S_true: (n_cal,) true-label scores S_i = s(X_i, Y_i)
    Returns:
      Z: (K, M) quantile embedding per class
      n_y: (K,) calibration counts per class
    """
    S_true = np.asarray(S_true, dtype=np.float64)
    y_cal = np.asarray(y_cal, dtype=int)
    M = int(q_grid.size)

    Z = np.zeros((K, M), dtype=np.float64)
    n_y = np.zeros(K, dtype=int)

    # global fallback embedding (used when n_y=0)
    global_q = np.quantile(S_true, q_grid, method="linear")

    for y in range(K):
        m = (y_cal == y)
        n_y[y] = int(m.sum())
        if n_y[y] <= 0:
            Z[y, :] = global_q
        else:
            Z[y, :] = np.quantile(S_true[m], q_grid, method="linear")

    return Z, n_y


def kmeans_cluster_classes(Z: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """
    Z: (K, M) embedding
    returns c_y: (K,) cluster assignment in {0,...,n_clusters-1}
    """
    Z = np.asarray(Z, dtype=np.float64)
    K = Z.shape[0]
    if n_clusters <= 0 or n_clusters > K:
        raise ValueError(f"n_clusters must be in [1, K]. Got {n_clusters}, K={K}")

    km = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init="auto")
    c_y = km.fit_predict(Z).astype(int)
    return c_y


# ============================================================
# Build classwise + clusterwise sorted arrays of true-label scores
# ============================================================

@dataclass
class SCCPIndex:
    # classwise sorted S_true per class
    s_class_sorted: List[np.ndarray]   # len K, each (n_y,)
    # clusterwise sorted S_true per cluster
    s_cluster_sorted: List[np.ndarray] # len n_clusters, each (N_k,)
    # calibration counts
    n_y: np.ndarray                    # (K,)
    N_k: np.ndarray                    # (n_clusters,)
    # class->cluster map
    c_y: np.ndarray                    # (K,)
    # metadata
    K: int
    n_clusters: int


def build_sccp_index(
    scores_cal: np.ndarray,  # (n_cal, K)
    y_cal: np.ndarray,       # (n_cal,)
    c_y: np.ndarray,         # (K,)
    n_clusters: int,
) -> SCCPIndex:
    scores_cal = np.asarray(scores_cal, dtype=np.float64)
    y_cal = np.asarray(y_cal, dtype=int)
    c_y = np.asarray(c_y, dtype=int)

    n_cal, K = scores_cal.shape
    if c_y.shape[0] != K:
        raise ValueError(f"c_y shape mismatch: {c_y.shape} vs K={K}")

    # true-label scores
    S_true = scores_cal[np.arange(n_cal), y_cal]  # (n_cal,)

    # classwise
    s_class_sorted: List[np.ndarray] = []
    n_y = np.zeros(K, dtype=int)
    for y in range(K):
        m = (y_cal == y)
        n_y[y] = int(m.sum())
        s_class_sorted.append(np.sort(S_true[m], axis=0))

    # clusterwise pooling by true label's cluster
    s_cluster_sorted: List[np.ndarray] = []
    N_k = np.zeros(n_clusters, dtype=int)
    for k in range(n_clusters):
        # cal points with label in cluster k
        m = (c_y[y_cal] == k)
        N_k[k] = int(m.sum())
        s_cluster_sorted.append(np.sort(S_true[m], axis=0))

    return SCCPIndex(
        s_class_sorted=s_class_sorted,
        s_cluster_sorted=s_cluster_sorted,
        n_y=n_y,
        N_k=N_k,
        c_y=c_y,
        K=K,
        n_clusters=n_clusters,
    )


# ============================================================
# Baseline LCCP: p_class(x,y) = u_class_ge(s(x,y)); include if > alpha
# ============================================================

def predict_sets_lccp(scores_X: np.ndarray, idx: SCCPIndex, alpha: float) -> np.ndarray:
    scores_X = np.asarray(scores_X, dtype=np.float64)
    n, K = scores_X.shape
    S = np.zeros((n, K), dtype=bool)
    for y in range(K):
        p_class = _u_from_sorted_ge(idx.s_class_sorted[y], scores_X[:, y])
        S[:, y] = (p_class > float(alpha))
    return S

# ============================================================
# CCCP : p_cluster(x,y) = u_cluster_ge(s(x,y)); include if > alpha
# ============================================================

def predict_sets_cccp(scores_X: np.ndarray, idx: SCCPIndex, alpha: float) -> np.ndarray:
    """
    CCCP-style baseline: cluster-only pooling
    p_cluster(x,y) = u_cluster_ge(s(x,y)); include if > alpha
    """
    scores_X = np.asarray(scores_X, dtype=np.float64)
    n, K = scores_X.shape
    S = np.zeros((n, K), dtype=bool)
    for y in range(K):
        k = int(idx.c_y[y])
        p_cluster = _u_from_sorted_ge(idx.s_cluster_sorted[k], scores_X[:, y])
        S[:, y] = (p_cluster > float(alpha))
    return S

# ============================================================
# Proposed SCCP: class-cluster shrinkage on p-values
#
# p_mix(x,y) = (1-lambda_y) * p_cluster(x,y) + lambda_y * p_class(x,y)
# lambda_y = n_y / (n_y + tau)
# include if p_mix > alpha
# ============================================================

def lambda_from_counts(n_y: np.ndarray, tau: float) -> np.ndarray:
    n_y = np.asarray(n_y, dtype=np.float64)
    tau = float(tau)
    return n_y / (n_y + tau)


def predict_sets_sccp_class_cluster(
    scores_X: np.ndarray,
    idx: SCCPIndex,
    tau: float,
    alpha: float,
) -> np.ndarray:
    scores_X = np.asarray(scores_X, dtype=np.float64)
    n, K = scores_X.shape

    lam = lambda_from_counts(idx.n_y, tau=float(tau))  # (K,)

    S = np.zeros((n, K), dtype=bool)
    for y in range(K):
        k = int(idx.c_y[y])
        p_class = _u_from_sorted_ge(idx.s_class_sorted[y], scores_X[:, y])
        p_cluster = _u_from_sorted_ge(idx.s_cluster_sorted[k], scores_X[:, y])

        # logit mixing
        eps = 1e-12
        pc = np.clip(p_cluster, eps, 1 - eps)
        py = np.clip(p_class,   eps, 1 - eps)

        logit = lambda p: np.log(p) - np.log(1 - p)
        sigmoid = lambda u: 1.0 / (1.0 + np.exp(-u))

        p_mix = sigmoid((1.0 - lam[y]) * logit(pc) + lam[y] * logit(py))
        
        S[:, y] = (p_mix > float(alpha))
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
# Main: build clustering on CAL; tune tau on SEL; evaluate on TEST
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--K", type=int, required=True, help="number of classes")
    ap.add_argument("--alpha", type=float, default=0.1)

    # quantile embedding + clustering
    ap.add_argument("--M", type=int, default=25, help="number of quantiles for embedding")
    ap.add_argument("--n_clusters", type=int, default=50, help="k-means clusters for classes")
    ap.add_argument("--cluster_seed", type=int, default=1)

    # tau tuning for lambda_y = n_y/(n_y+tau)
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

    # nonconformity scores s(x,y) = 1 - p_y(x)
    s_sel = scores_softmax(P_sel)
    s_cal = scores_softmax(P_cal)
    s_test = scores_softmax(P_test)

    # tail set (for reporting)
    tail_set = maybe_load_tail_set(args.npz, K=args.K, tail_frac=args.tail_frac)
    if tail_set.size == 0:
        y_all = np.concatenate([y_sel, y_cal, y_test], axis=0)
        counts = np.bincount(y_all, minlength=args.K)
        tail_set = tail_from_counts(counts, tail_frac=args.tail_frac)

    # -------------------------
    # Build class embedding on CAL (using true-label calibration scores)
    # -------------------------
    S_true_cal = s_cal[np.arange(s_cal.shape[0]), y_cal]
    q_grid = make_quantile_grid(args.M)
    Z, n_y_cal = class_quantile_embedding(S_true_cal, y_cal, K=args.K, q_grid=q_grid)

    # k-means clustering over classes
    c_y = kmeans_cluster_classes(Z, n_clusters=int(args.n_clusters), seed=int(args.cluster_seed))

    # Build indices (classwise + clusterwise sorted arrays)
    idx = build_sccp_index(s_cal, y_cal, c_y=c_y, n_clusters=int(args.n_clusters))

    # ---- TEST evaluation: LCCP / CCCP / SCCP
    S_lccp_test = predict_sets_lccp(s_test, idx, alpha=args.alpha)
    met_lccp = eval_metrics(S_lccp_test, y_test, alpha=args.alpha, tail_set=tail_set)

    S_cccp_test = predict_sets_cccp(s_test, idx, alpha=args.alpha)
    met_cccp = eval_metrics(S_cccp_test, y_test, alpha=args.alpha, tail_set=tail_set)

    print(f"[file] {args.npz}")
    print(f"[K classes] {args.K}  [alpha] {args.alpha}")
    print(f"[embedding] M={args.M}  [kmeans] n_clusters={args.n_clusters} seed={args.cluster_seed}")
    print(f"[tail] frac={args.tail_frac}  m={len(tail_set)}")
    print("")
    print("=== Baseline: LCCP (classwise) on TEST ===")
    print(json.dumps(met_lccp, indent=2))
    print("")

    print("=== Baseline: CCCP (clusterwise) on TEST ===")
    print(json.dumps(met_cccp, indent=2))
    print("")

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

    if args.print_tau_table:
        print("tau | sel_marg_cov | sel_cov_tail | sel_worst | sel_avg_size | sel_tail_size | feasible")
        print("-" * 86)

    for tau in tau_grid:
        S_sel = predict_sets_sccp_class_cluster(s_sel, idx, tau=float(tau), alpha=args.alpha)
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
            target = 1.0 - float(args.alpha)
            obj = (
                abs(met_sel["avg_class_cov"] - target),
                met_sel["avg_size"],
                -met_sel["worst_class_cov"],
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
            S_sel = predict_sets_sccp_class_cluster(s_sel, idx, tau=float(tau), alpha=args.alpha)
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

    # ---- Evaluate on TEST with best tau
    S_sccp_test = predict_sets_sccp_class_cluster(s_test, idx, tau=float(best_tau), alpha=args.alpha)
    met_sccp = eval_metrics(S_sccp_test, y_test, alpha=args.alpha, tail_set=tail_set)

    print("=== Proposed: SCCP (class--cluster shrinkage) on TEST ===")
    print(json.dumps(met_sccp, indent=2))
    print("")

    # Optional save
    if args.out_json:
        out = {
        "npz": args.npz,
        "K_classes": int(args.K),
        "alpha": float(args.alpha),
        "tail_frac": float(args.tail_frac),
        "tail_set": tail_set.tolist(),

        "score": {
            "name": "softmax_nonconformity",
            "definition": "s(x,y)=1-p_y(x)",
        },

        "embedding": {
            "method": "quantile_embedding",
            "M": int(args.M),
            "q_grid": q_grid.tolist(),
        },

        "clustering": {
            "method": "kmeans",
            "n_clusters": int(args.n_clusters),
            "seed": int(args.cluster_seed),
            "c_y": c_y.tolist(),  # class -> cluster
        },

        "methods": {
            "LCCP_classwise": {
                "description": "classwise conformal p-values (no pooling)",
                "test_metrics": met_lccp,   
            },
            "CCCP_clusterwise": {
                "description": "clusterwise pooling only (CCCP-style baseline)",
                "test_metrics": met_cccp,  
            },
            "SCCP_class_cluster": {
                "description": "class--cluster shrinkage on p-values",
                "shrinkage": {
                    "lambda_y": "n_y / (n_y + tau)",
                    "mixing": "p_mix = (1-lambda_y)*p_cluster + lambda_y*p_class",
                },
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
                "test_metrics": met_sccp,   # <-- SCCP TEST metrics로 (met_prop 말고 met_sccp 추천)
            },
        },
    }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()