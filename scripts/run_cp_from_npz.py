#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from collections import Counter

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


# ============================================================
# NPZ loading utilities (sel / cal / test) + optional logits
# ============================================================

def _is_prob_matrix(a: np.ndarray, K: int) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == K and np.isfinite(a).all()

def _is_label_vector(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 1 and np.issubdtype(a.dtype, np.integer)

def _is_logit_matrix(a: np.ndarray, K: int) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == K and np.isfinite(a).all()

def _normalize_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = P.sum(axis=1, keepdims=True)
    s = np.where(s <= eps, 1.0, s)
    Pn = P / s
    Pn = np.clip(Pn, eps, 1.0)
    Pn = Pn / Pn.sum(axis=1, keepdims=True)
    return Pn

def _find_by_name(d: Dict[str, np.ndarray], include: Tuple[str, ...], K: int, kind: str = "prob") -> Optional[np.ndarray]:
    for k in d.keys():
        lk = k.lower()
        if all(s in lk for s in include):
            a = d[k]
            if kind == "prob" and _is_prob_matrix(a, K):
                return a
            if kind == "logit" and _is_logit_matrix(a, K):
                return a
    return None

def _find_label_by_name(d: Dict[str, np.ndarray], include: Tuple[str, ...]) -> Optional[np.ndarray]:
    for k in d.keys():
        lk = k.lower()
        if all(s in lk for s in include):
            a = d[k]
            if _is_label_vector(a):
                return a
    return None

def load_npz_splits(
    path: str, K: int, fallback_split_seed: int = 1
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    raw = np.load(path, allow_pickle=True)
    d = {k: raw[k] for k in raw.files}

    def _pick_first_nonnull(items):
        for x in items:
            if x is not None:
                return x
        return None

    P_sel = _pick_first_nonnull([
        _find_by_name(d, ("p", "sel"), K, "prob"),
        _find_by_name(d, ("prob", "sel"), K, "prob"),
        _find_by_name(d, ("probs", "sel"), K, "prob"),
    ])
    y_sel = _pick_first_nonnull([
        _find_label_by_name(d, ("y", "sel")),
        _find_label_by_name(d, ("label", "sel")),
        _find_label_by_name(d, ("labels", "sel")),
    ])

    P_cal = _pick_first_nonnull([
        _find_by_name(d, ("p", "cal"), K, "prob"),
        _find_by_name(d, ("prob", "cal"), K, "prob"),
        _find_by_name(d, ("probs", "cal"), K, "prob"),
    ])
    y_cal = _pick_first_nonnull([
        _find_label_by_name(d, ("y", "cal")),
        _find_label_by_name(d, ("label", "cal")),
        _find_label_by_name(d, ("labels", "cal")),
    ])

    P_test = _pick_first_nonnull([
        _find_by_name(d, ("p", "test"), K, "prob"),
        _find_by_name(d, ("prob", "test"), K, "prob"),
        _find_by_name(d, ("probs", "test"), K, "prob"),
        _find_by_name(d, ("p", "val"), K, "prob"),
        _find_by_name(d, ("prob", "val"), K, "prob"),
        _find_by_name(d, ("probs", "val"), K, "prob"),
    ])
    y_test = _pick_first_nonnull([
        _find_label_by_name(d, ("y", "test")),
        _find_label_by_name(d, ("label", "test")),
        _find_label_by_name(d, ("labels", "test")),
        _find_label_by_name(d, ("y", "val")),
        _find_label_by_name(d, ("label", "val")),
        _find_label_by_name(d, ("labels", "val")),
    ])

    # fallback: length matching
    prob_keys = [k for k, v in d.items() if _is_prob_matrix(v, K)]
    lab_keys = [k for k, v in d.items() if _is_label_vector(v)]
    probs = {k: d[k] for k in prob_keys}
    labs = {k: d[k] for k in lab_keys}

    def match_prob_label(Pcand: Optional[np.ndarray], ycand: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if Pcand is None:
            return None
        if ycand is not None and len(ycand) == Pcand.shape[0]:
            return Pcand, ycand
        for _, yv in labs.items():
            if len(yv) == Pcand.shape[0]:
                return Pcand, yv
        return None

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tag, Pc, yc in [("sel", P_sel, y_sel), ("cal", P_cal, y_cal), ("test", P_test, y_test)]:
        m = match_prob_label(Pc, yc)
        if m is not None:
            splits[tag] = m

    if "cal" not in splits:
        candidates = []
        for _, Pv in probs.items():
            my = None
            for _, yv in labs.items():
                if len(yv) == Pv.shape[0]:
                    my = yv
                    break
            if my is not None:
                candidates.append((Pv, my))
        candidates.sort(key=lambda t: t[0].shape[0], reverse=True)
        if len(candidates) == 0:
            raise RuntimeError("Could not find any probs+labels pair in NPZ.")
        splits["cal"] = candidates[min(1, len(candidates) - 1)]

    if "test" not in splits:
        P, y = splits["cal"]
        rng = np.random.default_rng(fallback_split_seed)
        idx = rng.permutation(P.shape[0])
        n_cal = P.shape[0] // 2
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]
        splits["cal"] = (P[cal_idx], y[cal_idx])
        splits["test"] = (P[test_idx], y[test_idx])

    if "sel" not in splits:
        splits["sel"] = splits["cal"]

    # normalize
    for k in list(splits.keys()):
        P, y = splits[k]
        P = _normalize_rows(P.astype(np.float64))
        y = y.astype(int)
        splits[k] = (P, y)

    return splits

from typing import Dict, Optional, Tuple
import numpy as np

def load_npz_logits(path: str, K: int) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    d = {k: raw[k] for k in raw.files}

    def pick(tag: str) -> np.ndarray:
        cand_list = [
            _find_by_name(d, ("z", tag), K, kind="logit"),        # z_sel, z_cal, z_test
            _find_by_name(d, ("logit", tag), K, kind="logit"),    
            _find_by_name(d, ("zlogit", tag), K, kind="logit"),
        ]
        for x in cand_list:
            if x is not None:
                return x.astype(np.float64)
        raise RuntimeError(f"Could not find logits for tag='{tag}' in NPZ. keys={list(d.keys())}")

    return {"sel": pick("sel"), "cal": pick("cal"), "test": pick("test")}


# ============================================================
# Tail / head helper
# ============================================================

def tail_from_counts(counts_pool: np.ndarray, K: int, tail_frac: float) -> np.ndarray:
    counts_pool = np.asarray(counts_pool, dtype=float)
    if counts_pool.shape[0] != K:
        raise ValueError(f"counts_pool length {counts_pool.shape[0]} != K={K}")
    m = int(np.ceil(float(tail_frac) * K))
    m = max(0, min(m, K))
    order = np.argsort(counts_pool)  # ascending
    return order[:m].astype(int)

def build_tail_set_from_npz(z_npz: np.lib.npyio.NpzFile, K: int, tail_frac: float, tail_mode: str) -> np.ndarray:
    has_tail = ("tail_set" in z_npz.files)
    has_counts = ("counts_pool" in z_npz.files)

    if tail_mode == "npz":
        if has_tail:
            return np.asarray(z_npz["tail_set"], dtype=int)
        if has_counts:
            return tail_from_counts(z_npz["counts_pool"], K=K, tail_frac=tail_frac)
        raise ValueError("Need tail_set or counts_pool in NPZ.")

    if tail_mode == "counts_pool":
        if not has_counts:
            raise ValueError("tail_mode='counts_pool' requires counts_pool in NPZ.")
        return tail_from_counts(z_npz["counts_pool"], K=K, tail_frac=tail_frac)

    if tail_mode == "override":
        if not has_counts:
            raise ValueError("tail_mode='override' requires counts_pool in NPZ.")
        return tail_from_counts(z_npz["counts_pool"], K=K, tail_frac=tail_frac)

    raise ValueError(f"Unknown tail_mode={tail_mode}")


# ============================================================
# Scores (softmax / APS / RAPS)
# ============================================================

def scores_all_softmax(P: np.ndarray) -> np.ndarray:
    return 1.0 - P

def scores_all_APS(P: np.ndarray, randomize: bool = True, seed: int = 0) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    n, K = P.shape

    order = np.argsort(-P, axis=1)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cumsum = np.cumsum(P_sorted, axis=1)

    inv = np.empty_like(order)
    inv[np.arange(n)[:, None], order] = np.arange(K)[None, :]
    cum_at_label = cumsum[np.arange(n)[:, None], inv]

    if not randomize:
        return cum_at_label - P
    rng = np.random.default_rng(seed)
    U = rng.random(size=P.shape)
    return cum_at_label - U * P

def scores_all_RAPS(P: np.ndarray, lmbda: float, kreg: int, randomize: bool = True, seed: int = 0) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    n, K = P.shape

    order = np.argsort(-P, axis=1)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cumsum = np.cumsum(P_sorted, axis=1)

    inv = np.empty_like(order)
    inv[np.arange(n)[:, None], order] = np.arange(K)[None, :]
    cum_at_label = cumsum[np.arange(n)[:, None], inv]

    rank = inv + 1
    reg = np.maximum(lmbda * (rank - int(kreg)), 0.0)
    base = cum_at_label + reg

    if not randomize:
        return base - P
    rng = np.random.default_rng(seed)
    U = rng.random(size=P.shape)
    return base - U * P

def get_scores_all(P: np.ndarray, score: str, seed: int, raps_lambda: float, raps_kreg: int, randomize: bool) -> np.ndarray:
    score = score.lower()
    if score == "softmax":
        return scores_all_softmax(P)
    if score == "aps":
        return scores_all_APS(P, randomize=randomize, seed=seed)
    if score == "raps":
        return scores_all_RAPS(P, lmbda=raps_lambda, kreg=raps_kreg, randomize=randomize, seed=seed)
    raise ValueError(f"Unknown score='{score}' (choose softmax/aps/raps)")


# ============================================================
# Conformal quantile utilities
# ============================================================

def quantile_upper_conservative(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return float(np.inf)
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(scores)[k])


# ============================================================
# CCCP
# ============================================================

def get_quantile_threshold(alpha: float) -> int:
    n = 1
    while np.ceil((n + 1) * (1 - alpha) / n) > 1:
        n += 1
    return n

def get_conformal_quantile_ding(scores_true: np.ndarray, alpha: float, default_qhat: float = np.inf) -> float:
    scores_true = np.asarray(scores_true, dtype=float)
    n = len(scores_true)
    if n == 0:
        return float(default_qhat)
    val = np.ceil((n + 1) * (1 - alpha)) / n
    if val > 1:
        return float(default_qhat)
    return float(np.quantile(scores_true, val, method="inverted_cdf"))

def embed_all_classes_ding(scores_true: np.ndarray, labels: np.ndarray, K: int, q=(0.5, 0.6, 0.7, 0.8, 0.9)) -> Tuple[np.ndarray, np.ndarray]:
    scores_true = np.asarray(scores_true, dtype=float)
    labels = np.asarray(labels, dtype=int)
    q = np.asarray(q, dtype=float)

    emb = np.zeros((K, len(q)), dtype=float)
    cts = np.zeros((K,), dtype=float)

    for k in range(K):
        sk = scores_true[labels == k]
        cts[k] = sk.shape[0]
        if sk.shape[0] == 0:
            emb[k] = np.nan
        else:
            emb[k] = np.quantile(sk, q, method="inverted_cdf")

    if np.any(~np.isfinite(emb)):
        g = np.quantile(scores_true, q, method="inverted_cdf") if len(scores_true) else np.zeros(len(q))
        bad = ~np.isfinite(emb)
        emb[bad] = np.take(g, np.where(bad)[1])

    return emb, cts

def cccp_thresholds_ding(
    scores_all_totalcal: np.ndarray,
    y_totalcal: np.ndarray,
    alpha: float,
    frac_clustering: float,
    num_clusters: int,
    seed: int,
    embed_q=(0.5, 0.6, 0.7, 0.8, 0.9),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)

    scores_all_totalcal = np.asarray(scores_all_totalcal, dtype=float)
    y_totalcal = np.asarray(y_totalcal, dtype=int)
    n, K = scores_all_totalcal.shape

    scores_true = scores_all_totalcal[np.arange(n), y_totalcal]

    idx1 = rng.random(n) < float(frac_clustering)
    scores1, labels1 = scores_true[idx1], y_totalcal[idx1]
    scores2, labels2 = scores_true[~idx1], y_totalcal[~idx1]
    if len(labels2) == 0:
        scores2, labels2 = scores_true, y_totalcal

    thresh = get_quantile_threshold(alpha)
    cts1 = Counter(labels1.tolist())
    rare_classes = np.array([k for k in range(K) if cts1.get(k, 0) < thresh], dtype=int)
    nonrare = np.setdiff1d(np.arange(K), rare_classes)

    class2cluster = -np.ones((K,), dtype=int)

    clustering_done = False
    if (KMeans is not None) and (len(nonrare) > max(int(num_clusters), 1)) and (int(num_clusters) > 1):
        emb, cts = embed_all_classes_ding(scores1, labels1, K=K, q=embed_q)
        X = emb[nonrare]
        w = np.sqrt(np.maximum(cts[nonrare], 0.0))

        km = KMeans(n_clusters=int(num_clusters), random_state=seed, n_init=10)
        km.fit(X, sample_weight=w)
        class2cluster[nonrare] = km.labels_
        clustering_done = True

    null_qhat = get_conformal_quantile_ding(scores2, alpha, default_qhat=np.inf)
    t_class = np.full((K,), null_qhat, dtype=float)

    if clustering_done:
        clusters2 = class2cluster[labels2]
        C = int(class2cluster[nonrare].max()) + 1 if len(nonrare) else 0

        cluster_qhat = np.full((C,), np.inf, dtype=float)
        for c in range(C):
            sc = scores2[clusters2 == c]
            cluster_qhat[c] = get_conformal_quantile_ding(sc, alpha, default_qhat=np.inf)

        for k in nonrare:
            ck = class2cluster[k]
            if ck >= 0:
                t_class[k] = cluster_qhat[ck]

    info = {
        "thresh": float(thresh),
        "num_rare": float(len(rare_classes)),
        "gamma": float(frac_clustering),
        "M": float(num_clusters),
        "clustering_done": float(clustering_done),
    }
    return t_class, class2cluster, info


# ============================================================
# SCCP utilities
# ============================================================

def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def class_quantile_embedding_from_vector(
    v_true: np.ndarray, y: np.ndarray, K: int, q_grid: np.ndarray, fallback_global: bool = True
) -> np.ndarray:
    """
    v_true: (n,) scalar per sample (e.g., true-label score, or -true-logit)
    y: (n,)
    Returns emb: (K, len(q_grid))
    """
    q_grid = np.asarray(q_grid, dtype=np.float64)
    emb = np.zeros((K, len(q_grid)), dtype=np.float64)

    if fallback_global:
        gq = np.quantile(v_true, q_grid, method="higher") if len(v_true) else np.zeros(len(q_grid))
    else:
        gq = np.zeros(len(q_grid))

    for k in range(K):
        mk = (y == k)
        if mk.any():
            emb[k] = np.quantile(v_true[mk], q_grid, method="higher")
        else:
            emb[k] = gq

    bad = ~np.isfinite(emb)
    if bad.any():
        emb[bad] = np.take(gq, np.where(bad)[1])
    return emb

def kmeans_labels_on_classes(
    X_class: np.ndarray,  # (K, d)
    n_clusters: int,
    seed: int,
    class_weight: Optional[np.ndarray] = None,  # (K,)
    weighted_kmeans: bool = True,
) -> np.ndarray:
    K = X_class.shape[0]
    n_clusters = max(1, min(int(n_clusters), K))

    if (KMeans is not None) and weighted_kmeans and (n_clusters > 1):
        w = None
        if class_weight is not None:
            w = np.asarray(class_weight, dtype=float)
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        km.fit(X_class, sample_weight=w)
        return km.labels_.astype(int)

    # fallback: unweighted simple kmeans
    rng = np.random.default_rng(seed)
    centers = X_class[rng.choice(K, size=n_clusters, replace=False)].copy()
    labels = np.zeros(K, dtype=int)
    for _ in range(50):
        d2 = ((X_class[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for c in range(n_clusters):
            m = (labels == c)
            if m.any():
                centers[c] = X_class[m].mean(axis=0)
            else:
                centers[c] = X_class[rng.integers(0, K)]
    return labels.astype(int)

def sccp_fit_thresholds(
    scores_all_D2: np.ndarray, y_D2: np.ndarray,
    class2cluster: np.ndarray,
    alpha: float,
    tau: float,
    beta: float,
) -> np.ndarray:
    """
    Compute per-class thresholds using:
      - global threshold on D2
      - cluster threshold on D2
      - per-class shrinkage weight tau_y = tau/(tau + n_y^beta)
        and mix cluster->global (simple)
    """
    y_D2 = np.asarray(y_D2, dtype=int)
    n2, K = scores_all_D2.shape
    scores_true_D2 = scores_all_D2[np.arange(n2), y_D2]

    t_global = quantile_upper_conservative(scores_true_D2, alpha)

    C = int(np.max(class2cluster)) + 1
    t_cluster = np.full((C,), t_global, dtype=float)
    n_cluster = np.zeros((C,), dtype=int)

    for c in range(C):
        cls_in_c = np.where(class2cluster == c)[0]
        m = np.isin(y_D2, cls_in_c)
        n_cluster[c] = int(m.sum())
        if m.any():
            t_cluster[c] = quantile_upper_conservative(scores_true_D2[m], alpha)
        else:
            t_cluster[c] = t_global

    # optional: cluster->global shrinkage via same tau, using n_cluster
    t_mix_cluster = np.zeros((C,), dtype=float)
    for c in range(C):
        nc = float(n_cluster[c])
        wc = nc / (nc + tau) if (nc + tau) > 0 else 0.0
        t_mix_cluster[c] = wc * t_cluster[c] + (1.0 - wc) * t_global

    counts = np.bincount(y_D2, minlength=K).astype(float)
    t_class = np.zeros((K,), dtype=float)

    for y in range(K):
        ny = float(counts[y])
        denom = tau + (ny ** float(beta))
        tau_y = (tau / denom) if denom > 0 else 1.0  # in [0,1]
        cy = int(class2cluster[y])
        t_class[y] = (1.0 - tau_y) * t_mix_cluster[cy] + tau_y * t_global

    return t_class


# ============================================================
# Evaluation
# ============================================================

def eval_metrics(
    scores_all: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    alpha: float,
    tail_set: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    n, K = scores_all.shape

    t = np.asarray(t, dtype=float)
    S = (scores_all <= t[None, :])

    hit = S[np.arange(n), y].astype(float)
    set_sizes = S.sum(axis=1).astype(float)

    cov_k = np.full(K, np.nan, dtype=float)
    n_k = np.zeros(K, dtype=int)
    for k in range(K):
        m = (y == k)
        n_k[k] = int(m.sum())
        if n_k[k] > 0:
            cov_k[k] = float(np.mean(hit[m]))

    covgap = float(np.nanmean(np.abs(cov_k - (1.0 - alpha))))
    maxgap = float(np.nanmax(np.abs(cov_k - (1.0 - alpha))))

    out = {
        "marginal_cov": float(np.mean(hit)),
        "avg_size": float(np.mean(set_sizes)),
        "covgap": covgap,
        "maxgap": maxgap,
        "avg_class_cov": float(np.nanmean(cov_k)),
        "worst_class_cov": float(np.nanmin(cov_k)),
        "std_class_cov": float(np.nanstd(cov_k)),
    }

    if tail_set is not None:
        tail_set = np.asarray(tail_set, dtype=int)
        is_tail = np.isin(y, tail_set)
        is_head = ~is_tail

        out.update({
            "n_tail": int(is_tail.sum()),
            "n_head": int(is_head.sum()),
            "cov_tail": float(np.mean(hit[is_tail])) if is_tail.any() else float("nan"),
            "cov_head": float(np.mean(hit[is_head])) if is_head.any() else float("nan"),
            "size_tail": float(np.mean(set_sizes[is_tail])) if is_tail.any() else float("nan"),
            "size_head": float(np.mean(set_sizes[is_head])) if is_head.any() else float("nan"),
        })

    return out


# ============================================================
# Printing helpers
# ============================================================

@dataclass
class Row:
    method: str
    marginal_cov: float
    avg_size: float
    covgap: float
    cov_tail: float = float("nan")
    size_tail: float = float("nan")
    cov_head: float = float("nan")
    size_head: float = float("nan")

def fmt_row(r: Row) -> str:
    return (
        f"{r.method:10s} | {r.marginal_cov:7.4f} | {r.avg_size:9.2f} | {r.covgap:7.4f} | "
        f"{r.cov_tail:7.4f} | {r.size_tail:8.2f} | {r.cov_head:7.4f} | {r.size_head:8.2f}"
    )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)

    # Score choice
    ap.add_argument("--score", type=str, default="softmax", choices=["softmax", "aps", "raps"])
    ap.add_argument("--scores", type=str, default="", help="Comma-separated list among softmax,aps,raps. If set, overrides --score and runs all.")
    ap.add_argument("--no_randomize", action="store_true")
    ap.add_argument("--raps_lambda", type=float, default=0.0)
    ap.add_argument("--raps_kreg", type=int, default=1)

    # SCCP hyperparams (fixed-run fallback)
    ap.add_argument("--clusters", type=int, default=10)
    ap.add_argument("--tau", type=float, default=50.0)
    ap.add_argument("--beta", type=float, default=0.5, help="Shrinkage exponent beta in tau/(tau + n_y^beta). Default 0.5.")

    ap.add_argument("--q_grid", type=str, default=None, help="Comma-separated quantiles for SCCP embedding. default includes 1-alpha.")
    ap.add_argument("--emb_source", type=str, default="logit", choices=["logit", "score"],
                    help="Embedding source for clustering: logit uses -true_logit (needs logits in npz), score uses true-label score from chosen score.")
    ap.add_argument("--weighted_kmeans", action="store_true", help="Use sklearn KMeans with class weights ~ sqrt(class count) (D1).")

    # Data-dependent selection via calibration split
    ap.add_argument("--use_calib_split", action="store_true",
                    help="Split calibration into D1/D2/D3; tune (Kc,tau) by minimizing covgap on D3.")
    ap.add_argument("--calib_fracs", type=str, default="0.33,0.34,0.33",
                    help="Fractions for D1,D2,D3 split of calibration. e.g. '0.33,0.34,0.33'")
    ap.add_argument("--kc_grid", type=str, default="10", help="Comma-separated Kc candidates for tuning.")
    ap.add_argument("--tau_grid", type=str, default="50", help="Comma-separated tau candidates for tuning.")

    # CCCP(Ding) optional
    ap.add_argument("--run_cccp", action="store_true", help="Also run Ding CCCP for comparison.")
    ap.add_argument("--cccp_gamma", type=float, default=0.5)
    ap.add_argument("--cccp_M", type=int, default=10)

    # Tail controls
    ap.add_argument("--tail_frac", type=float, default=0.2)
    ap.add_argument("--tail_mode", type=str, default="npz", choices=["npz", "counts_pool", "override"])

    ap.add_argument("--out", type=str, default="", help="Save json to path")

    args = ap.parse_args()
    randomize = (not args.no_randomize)

    # which scores to run
    if args.scores.strip():
        score_list = [s.strip().lower() for s in args.scores.split(",") if s.strip()]
        for s in score_list:
            if s not in ["softmax", "aps", "raps"]:
                raise ValueError(f"Invalid score in --scores: {s}")
    else:
        score_list = [args.score.lower()]

    # Load splits (probs)
    splits = load_npz_splits(args.npz, K=args.K, fallback_split_seed=args.seed)
    P_sel, y_sel = splits["sel"]
    P_cal, y_cal = splits["cal"]
    P_test, y_test = splits["test"]

    # Optional logits
    logits = load_npz_logits(args.npz, K=args.K)

    z_npz = np.load(args.npz, allow_pickle=True)
    tail_set = build_tail_set_from_npz(z_npz, K=args.K, tail_frac=args.tail_frac, tail_mode=args.tail_mode)

    # Parse q_grid
    q_grid = None
    if args.q_grid is not None:
        q_grid = parse_csv_floats(args.q_grid)
    if q_grid is None:
        q_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0 - float(args.alpha)]
    q_grid = np.asarray(q_grid, dtype=np.float64)

    # Parse calib split
    fracs = np.asarray(parse_csv_floats(args.calib_fracs), dtype=float)
    if fracs.shape[0] != 3 or np.any(fracs <= 0):
        raise ValueError("--calib_fracs must have 3 positive numbers, e.g. '0.33,0.34,0.33'")
    fracs = fracs / fracs.sum()

    kc_grid = parse_csv_ints(args.kc_grid)
    tau_grid = parse_csv_floats(args.tau_grid)
    if len(kc_grid) == 0 or len(tau_grid) == 0:
        raise ValueError("--kc_grid and --tau_grid must be non-empty.")

    rng = np.random.default_rng(args.seed)

    # Header
    print(f"[file] {args.npz}")
    print(f"[K] {args.K}  [alpha] {args.alpha}  [seed] {args.seed}")
    print(f"[tail] mode={args.tail_mode} tail_frac={args.tail_frac} m={len(tail_set)}")
    print(f"[SCCP] emb_source={args.emb_source} weighted_kmeans={args.weighted_kmeans} beta={args.beta}")
    if args.use_calib_split:
        print(f"[Cal split] D1,D2,D3 fracs={fracs.tolist()}  kc_grid={kc_grid} tau_grid={tau_grid}")
    print("")

    all_out = {"npz": args.npz, "K": args.K, "alpha": args.alpha, "seed": args.seed, "results": {}}

    # ============================================================
    # Run per score
    # ============================================================
    for score_name in score_list:
        # Compute score matrices
        scores_sel = get_scores_all(P_sel, score_name, seed=args.seed, raps_lambda=args.raps_lambda, raps_kreg=args.raps_kreg, randomize=randomize)
        scores_cal = get_scores_all(P_cal, score_name, seed=args.seed, raps_lambda=args.raps_lambda, raps_kreg=args.raps_kreg, randomize=randomize)
        scores_test = get_scores_all(P_test, score_name, seed=args.seed, raps_lambda=args.raps_lambda, raps_kreg=args.raps_kreg, randomize=randomize)

        # -------- GCP (global) threshold on cal
        scores_true_cal = scores_cal[np.arange(scores_cal.shape[0]), y_cal]
        t_global = quantile_upper_conservative(scores_true_cal, args.alpha)
        eg = eval_metrics(scores_test, y_test, np.full((args.K,), t_global), alpha=args.alpha, tail_set=tail_set)

        # -------- CCCP (optional)
        ec = None
        info_cccp = None
        if args.run_cccp:
            t_cccp, c2c_cccp, info_cccp = cccp_thresholds_ding(
                scores_all_totalcal=scores_cal,
                y_totalcal=y_cal,
                alpha=args.alpha,
                frac_clustering=args.cccp_gamma,
                num_clusters=args.cccp_M,
                seed=args.seed,
                embed_q=(0.5, 0.6, 0.7, 0.8, 0.9),
            )
            ec = eval_metrics(scores_test, y_test, t_cccp, alpha=args.alpha, tail_set=tail_set)

        # -------- SCCP (fixed or tuned)
        if args.use_calib_split:
            # Split calibration indices into D1/D2/D3
            n_cal = P_cal.shape[0]
            perm = rng.permutation(n_cal)
            n1 = int(round(fracs[0] * n_cal))
            n2 = int(round(fracs[1] * n_cal))
            n1 = min(max(n1, 1), n_cal - 2)
            n2 = min(max(n2, 1), n_cal - n1 - 1)
            idx1 = perm[:n1]
            idx2 = perm[n1:n1 + n2]
            idx3 = perm[n1 + n2:]

            y1, y2, y3 = y_cal[idx1], y_cal[idx2], y_cal[idx3]
            scores1_all = scores_cal[idx1]
            scores2_all = scores_cal[idx2]
            scores3_all = scores_cal[idx3]

            # D1 embedding source
            if args.emb_source == "logit":
                Z_cal = logits.get("cal", None)
                if Z_cal is None:
                    raise ValueError("emb_source=logit but NPZ has no logits for 'cal'. Add z_cal/logit_cal to NPZ.")
                z1 = Z_cal[idx1]
                # use nonconformity-like scalar: -true_logit
                v1_true = -z1[np.arange(z1.shape[0]), y1]
            else:
                # embedding from true-label score
                v1_true = scores1_all[np.arange(scores1_all.shape[0]), y1]

            emb = class_quantile_embedding_from_vector(v1_true, y1, K=args.K, q_grid=q_grid, fallback_global=True)

            # class weights: sqrt(class sample size on D1)
            counts1 = np.bincount(y1, minlength=args.K).astype(float)
            class_w = np.sqrt(np.maximum(counts1, 0.0))

            # grid search over (Kc, tau) using D3 covgap
            best = None
            best_tuple = None
            best_class2cluster = None
            best_t = None

            for Kc in kc_grid:
                c2c = kmeans_labels_on_classes(emb, n_clusters=Kc, seed=args.seed, class_weight=class_w, weighted_kmeans=args.weighted_kmeans)
                for tau in tau_grid:
                    t_cls = sccp_fit_thresholds(
                        scores_all_D2=scores2_all,
                        y_D2=y2,
                        class2cluster=c2c,
                        alpha=args.alpha,
                        tau=float(tau),
                        beta=float(args.beta),
                    )
                    ed3 = eval_metrics(scores3_all, y3, t_cls, alpha=args.alpha, tail_set=tail_set)
                    obj = ed3["covgap"]
                    # tie-break: smaller avg_size on D3
                    tie = ed3["avg_size"]
                    cand = (obj, tie)
                    if best is None or cand < best:
                        best = cand
                        best_tuple = (int(Kc), float(tau))
                        best_class2cluster = c2c.copy()
                        best_t = t_cls.copy()

            # evaluate chosen SCCP on TEST
            es = eval_metrics(scores_test, y_test, best_t, alpha=args.alpha, tail_set=tail_set)
            sccp_info = {"tuned": True, "best_kc": best_tuple[0], "best_tau": best_tuple[1], "beta": float(args.beta)}
        else:
            # fixed SCCP: clustering on SEL split (as before), thresholds on CAL split
            scores_true_sel = scores_sel[np.arange(scores_sel.shape[0]), y_sel]

            if args.emb_source == "logit":
                Z_sel = logits.get("sel", None)
                if Z_sel is None:
                    raise ValueError("emb_source=logit but NPZ has no logits for 'sel'. Add z_sel/logit_sel to NPZ.")
                v_true = -Z_sel[np.arange(Z_sel.shape[0]), y_sel]
            else:
                v_true = scores_true_sel

            emb = class_quantile_embedding_from_vector(v_true, y_sel, K=args.K, q_grid=q_grid, fallback_global=True)
            counts_sel = np.bincount(y_sel, minlength=args.K).astype(float)
            class_w = np.sqrt(np.maximum(counts_sel, 0.0))

            c2c = kmeans_labels_on_classes(emb, n_clusters=args.clusters, seed=args.seed, class_weight=class_w, weighted_kmeans=args.weighted_kmeans)
            t_sccp = sccp_fit_thresholds(scores_all_D2=scores_cal, y_D2=y_cal, class2cluster=c2c, alpha=args.alpha, tau=float(args.tau), beta=float(args.beta))
            es = eval_metrics(scores_test, y_test, t_sccp, alpha=args.alpha, tail_set=tail_set)
            sccp_info = {"tuned": False, "clusters": int(args.clusters), "tau": float(args.tau), "beta": float(args.beta)}

        # -------- print block
        print(f"==================== score={score_name} (randomize={randomize}) ====================")
        print("method      | marg_cov |  avg_size |  covgap | cov_tail |  sz_tail | cov_head |  sz_head")
        print("-" * 92)

        rows = [
            Row("GCP", eg["marginal_cov"], eg["avg_size"], eg["covgap"], eg.get("cov_tail", np.nan), eg.get("size_tail", np.nan), eg.get("cov_head", np.nan), eg.get("size_head", np.nan)),
        ]
        if ec is not None:
            rows.append(Row("CCCP(Ding)", ec["marginal_cov"], ec["avg_size"], ec["covgap"], ec.get("cov_tail", np.nan), ec.get("size_tail", np.nan), ec.get("cov_head", np.nan), ec.get("size_head", np.nan)))
        rows.append(Row("SCCP", es["marginal_cov"], es["avg_size"], es["covgap"], es.get("cov_tail", np.nan), es.get("size_tail", np.nan), es.get("cov_head", np.nan), es.get("size_head", np.nan)))

        for r in rows:
            print(fmt_row(r))
        print("")

        all_out["results"][score_name] = {
            "score": score_name,
            "randomize": randomize,
            "raps_lambda": float(args.raps_lambda),
            "raps_kreg": int(args.raps_kreg),
            "GCP": {"t_global": float(t_global), "metrics": eg},
            "SCCP": {"info": sccp_info, "metrics": es},
        }
        if ec is not None:
            all_out["results"][score_name]["CCCP"] = {"info": info_cccp, "metrics": ec}

    # Save json
    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_out, f, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
