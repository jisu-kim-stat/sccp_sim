import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans


# -----------------------------
# Utilities
# -----------------------------
def sha10(obj) -> str:
    """실험 설정 dict -> json -> sha1 해시(앞 10글자)"""
    s = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:10]


def ensure_dir(path: str) -> None:
    """폴더 없으면 생성"""
    os.makedirs(path, exist_ok=True)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """npz를 dict로 로드; meta_json은 numpy scalar일 수 있어 문자열로 변환"""
    d = dict(np.load(path, allow_pickle=True))
    if "meta_json" in d and isinstance(d["meta_json"], np.ndarray) and d["meta_json"].shape == ():
        d["meta_json"] = str(d["meta_json"].item())
    return d


def check_npz(d: Dict[str, np.ndarray]) -> None:
    """필수 키 존재 및 확률/라벨 범위 기본 점검"""
    required = ["p_cal", "p_test", "y_cal", "y_test"]
    for k in required:
        if k not in d:
            raise ValueError(f"NPZ missing key '{k}'")

    p_cal = d["p_cal"]
    p_test = d["p_test"]
    y_cal = d["y_cal"].astype(int)
    y_test = d["y_test"].astype(int)

    if p_cal.ndim != 2 or p_test.ndim != 2:
        raise ValueError("p_cal/p_test must be 2D arrays")
    if p_cal.shape[1] != p_test.shape[1]:
        raise ValueError("K mismatch between p_cal and p_test")

    K = p_cal.shape[1]
    if y_cal.min() < 0 or y_cal.max() >= K:
        raise ValueError("y_cal out of range")
    if y_test.min() < 0 or y_test.max() >= K:
        raise ValueError("y_test out of range")

    # probability sanity checks
    for name, p in [("p_cal", p_cal), ("p_test", p_test)]:
        if np.nanmin(p) < -1e-6 or np.nanmax(p) > 1 + 1e-6:
            raise ValueError(f"{name} has values outside [0,1]")
        row_sums = p.sum(axis=1)
        if not np.isfinite(row_sums).all():
            raise ValueError(f"{name} has non-finite rows")
        if np.mean(np.abs(row_sums - 1.0)) > 1e-2:
            print(f"[WARN] {name} row sums deviate from 1. mean|sum-1|={np.mean(np.abs(row_sums-1)):.4f}")


def softmax_score(p: np.ndarray) -> np.ndarray:
    """nonconformity score S = 1 - p"""
    return 1.0 - p


def sorted_ge_count(sorted_arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    sorted_arr ascending일 때, 각 x에 대해 #{sorted_arr >= x} 반환.
    """
    idx = np.searchsorted(sorted_arr, x, side="left")
    return (sorted_arr.size - idx).astype(np.int64)


# -----------------------------
# CCV helper: b_i by MC for gamma family
# -----------------------------
def _simulate_order_stats(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """B번 반복으로 Uniform(0,1)^n 표본을 뽑고 각 row를 정렬 -> order stats"""
    U = rng.random((B, n), dtype=np.float64)
    U.sort(axis=1)
    return U  # (B,n) ascending


def _joint_prob(U_ord: np.ndarray, b: np.ndarray) -> float:
    """P( for all i, U_(i) >= b_i )를 MC로 근사"""
    ok = (U_ord >= b[None, :]).all(axis=1)
    return float(ok.mean())


def _beta_ppf_vector(i_arr: np.ndarray, n: int, gamma: float) -> np.ndarray:
    """
    b_i(gamma) = Beta(i, n+1-i) 의 gamma-quantile 벡터.
    scipy 없으면 명확히 에러 내도록(잘못된 근사로 조용히 진행 방지).
    """
    try:
        from scipy.stats import beta as beta_dist
    except Exception as e:
        raise RuntimeError(
            "scipy가 필요합니다. `pip install scipy` 후 다시 실행하세요."
        ) from e

    a = i_arr.astype(np.float64)
    b = (n + 1 - i_arr).astype(np.float64)
    return beta_dist.ppf(gamma, a, b).astype(np.float64)


def precompute_b_table_gamma(
    n: int,
    delta: float,
    B: int,
    seed: int,
    max_iter: int = 30,
) -> Tuple[np.ndarray, float]:
    """
    gamma를 이진탐색으로 찾아서,
      P( forall i: U_(i) >= b_i(gamma) ) >= 1-delta
    를 만족하는 가장 큰 gamma 선택.
    """
    rng = np.random.default_rng(seed)
    U_ord = _simulate_order_stats(n, B, rng)
    i_arr = np.arange(1, n + 1, dtype=np.int64)

    target = 1.0 - delta
    lo, hi = 1e-6, 1 - 1e-6
    best_gamma = lo
    best_b = np.zeros(n, dtype=np.float64)

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        b_mid = _beta_ppf_vector(i_arr, n, mid)
        p_mid = _joint_prob(U_ord, b_mid)
        if p_mid >= target:
            best_gamma, best_b = mid, b_mid
            lo = mid
        else:
            hi = mid
    return best_b, best_gamma


@dataclass
class CCVCache:
    cache_dir: str
    delta: float
    mc_B: int
    seed: int

    def path(self, n: int) -> str:
        return os.path.join(
            self.cache_dir,
            f"ccv_b_n{n}_d{self.delta:.6f}_B{self.mc_B}_s{self.seed}.npz",
        )

    def get_b(self, n: int) -> Optional[np.ndarray]:
        fp = self.path(n)
        if not os.path.exists(fp):
            return None
        z = np.load(fp, allow_pickle=True)
        return z["b"].astype(np.float64)

    def set_b(self, n: int, b: np.ndarray, gamma: float) -> None:
        fp = self.path(n)
        ensure_dir(self.cache_dir)
        np.savez(
            fp,
            b=b.astype(np.float64),
            gamma=np.array(gamma, dtype=np.float64),
            delta=np.array(self.delta, dtype=np.float64),
            B=np.array(self.mc_B, dtype=np.int64),
            seed=np.array(self.seed, dtype=np.int64),
        )


def ccv_h_from_b(b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    h(t) = max{i/n : b_i <= t}. (없으면 0)
    b는 nondecreasing이어야 함.
    """
    n = b.size
    k = np.searchsorted(b, t, side="right")
    return (k / n).astype(np.float64)


# -----------------------------
# Label clustering (CCCP-style)
# -----------------------------
def build_label_embeddings_score_quantile(
    scores_cal: np.ndarray,      # (Ncal,) true-label scores
    y_cal: np.ndarray,           # (Ncal,) labels in {0,...,K-1}
    K: int,
    alpha: float,
    q_grid: Optional[Sequence[float]] = None,
    min_count: int = 5,
    null_value: float = np.nan,
) -> np.ndarray:
    """
    각 클래스 y마다 calibration 점들의 score 분포에서 quantile 벡터를 뽑아 임베딩 E[y,:]로 사용.
    E shape: (K, m) where m=len(q_grid).

    클래스 표본 수가 min_count 미만이면 E[y,:]는 null_value로 채움.
    """
    if q_grid is None:
        q_grid = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0 - alpha], dtype=np.float64)
    else:
        q_grid = np.asarray(q_grid, dtype=np.float64)
        if not np.any(np.isclose(q_grid, 1.0 - alpha)):
            q_grid = np.sort(np.append(q_grid, 1.0 - alpha))

    m = q_grid.size
    E = np.full((K, m), null_value, dtype=np.float64)

    for y in range(K):
        s_y = scores_cal[y_cal == y]
        if s_y.size >= min_count:
            # numpy 버전 호환: method / interpolation
            try:
                E[y, :] = np.quantile(s_y.astype(np.float64), q_grid, method="linear")
            except TypeError:
                E[y, :] = np.quantile(s_y.astype(np.float64), q_grid, interpolation="linear")
    return E


def cluster_labels_kmeans_cccp(
    E: np.ndarray,          # (K,d)
    counts: np.ndarray,     # (K,)
    Kc: int,
    seed: int,
    weighted: bool = True,
    min_count: int = 5,
    null_id: int = -1,
) -> np.ndarray:
    """
    CCCP-style label clustering with a null cluster.

    Returns:
      cluster_id: (K,), each y mapped to {null_id,0,1,...,Kc_eff-1}.
    """
    K = E.shape[0]
    counts = counts.astype(np.int64, copy=False)

    finite = np.isfinite(E).all(axis=1)
    eligible = (counts >= min_count) & finite
    idx = np.where(eligible)[0]

    cluster_id = np.full(K, null_id, dtype=np.int64)

    if idx.size == 0:
        return cluster_id

    Kc_eff = int(min(Kc, idx.size))
    if Kc_eff <= 0:
        return cluster_id

    E_fit = E[idx, :]

    km = KMeans(n_clusters=Kc_eff, random_state=seed, n_init="auto")

    if weighted:
        w = np.maximum(counts[idx], 1).astype(np.float64)
        try:
            km.fit(E_fit, sample_weight=w)
        except TypeError:
            km.fit(E_fit)
    else:
        km.fit(E_fit)

    cluster_id[idx] = km.labels_.astype(np.int64)
    return cluster_id


# -----------------------------
# Core CP computations
# -----------------------------
def make_candidates(p_test: np.ndarray, y_test: np.ndarray, topL: int) -> List[np.ndarray]:
    """각 test 점마다 후보 라벨 topL을 만들고, true label은 반드시 포함."""
    N, K = p_test.shape
    if topL <= 0 or topL >= K:
        return [np.arange(K, dtype=np.int64) for _ in range(N)]

    cand_list: List[np.ndarray] = []
    for j in range(N):
        top = np.argpartition(-p_test[j], topL - 1)[:topL]
        y = int(y_test[j])
        if y not in top:
            top = np.concatenate([top, np.array([y], dtype=np.int64)])
        cand_list.append(np.unique(top))
    return cand_list


def _prep_global_cluster_scores(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    cluster_id: np.ndarray,
    null_id: int = -1,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Precompute:
      - global_sorted: sorted ascending true-label scores over ALL calibration points
      - clu_sorted[c]: sorted ascending true-label scores for cluster c (c>=0)
      - n_c         : counts per cluster c (c>=0)

    Points whose labels map to null_id are excluded from cluster lists.
    """
    N, K = p_cal.shape

    s_true = softmax_score(p_cal[np.arange(N), y_cal]).astype(np.float64)
    global_sorted = np.sort(s_true)

    valid_clusters = cluster_id[cluster_id != null_id]
    Kc = int(valid_clusters.max() + 1) if valid_clusters.size > 0 else 0

    clu_sorted: List[np.ndarray] = [np.empty((0,), dtype=np.float64) for _ in range(Kc)]
    n_c = np.zeros(Kc, dtype=np.int64)

    if Kc == 0:
        return global_sorted, clu_sorted, n_c

    clu_of_point = cluster_id[y_cal]  # (N,)
    for c in range(Kc):
        idx = np.where(clu_of_point == c)[0]
        if idx.size > 0:
            vals = s_true[idx]
            clu_sorted[c] = np.sort(vals)
            n_c[c] = idx.size

    return global_sorted, clu_sorted, n_c


def _rank_from_sorted(sorted_arr: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Tail-rank p-value: (1 + #{sorted_arr >= s}) / (n + 1)."""
    n = sorted_arr.size
    if n == 0:
        return np.ones_like(s, dtype=np.float64)
    ge = sorted_ge_count(sorted_arr, s)
    return (1.0 + ge) / (n + 1.0)


def _loo_rank_by_pos(n: int) -> np.ndarray:
    """
    sorted length n에서 position k(0..n-1) 원소의 count_ge including self = n-k.
    LOO proxy: r_loo(k) = (n-k)/n
    """
    if n <= 1:
        return np.ones((n,), dtype=np.float64)
    k = np.arange(n, dtype=np.int64)
    return ((n - k) / n).astype(np.float64)


# -----------------------------
# Global CP (GCP)
# -----------------------------
def eval_gcp(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    cand_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    Ntest = p_test.shape[0]
    Ncal = p_cal.shape[0]

    s_cal_true = softmax_score(p_cal[np.arange(Ncal), y_cal]).astype(np.float64)
    s_cal_sorted = np.sort(s_cal_true)

    covered = np.zeros(Ntest, dtype=bool)
    set_sizes = np.zeros(Ntest, dtype=np.int64)

    for j in range(Ntest):
        cand = cand_list[j]
        y_true = int(y_test[j])

        s_xy = softmax_score(p_test[j, cand]).astype(np.float64)
        ge = sorted_ge_count(s_cal_sorted, s_xy)
        u = (1.0 + ge) / (s_cal_sorted.size + 1.0)

        keep = u > alpha
        set_sizes[j] = int(np.sum(keep))
        true_pos = int(np.where(cand == y_true)[0][0])
        covered[j] = bool(keep[true_pos])

    return covered, set_sizes


# -----------------------------
# Ours: Global+Cluster shrinkage + CCV
# -----------------------------
def eval_ours_shrinkage_global_cluster_ccv(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    cand_list: List[np.ndarray],
    cluster_id: np.ndarray,
    tau: float,
    ccv_cache: Optional["CCVCache"] = None,
    ccv_delta: float = 0.1,
    ccv_nmax: int = 5000,
    null_id: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    N, K = p_test.shape

    global_sorted, clu_sorted, n_c = _prep_global_cluster_scores(
        p_cal=p_cal, y_cal=y_cal, cluster_id=cluster_id, null_id=null_id
    )

    Ncal = p_cal.shape[0]
    cal_true_scores = softmax_score(p_cal[np.arange(Ncal), y_cal]).astype(np.float64)
    clu_of_point = cluster_id[y_cal]

    Kc = len(clu_sorted)
    T_cal_by_cluster: List[np.ndarray] = [np.empty((0,), dtype=np.float64) for _ in range(Kc)]

    # global LOO proxy
    pos_global = np.searchsorted(global_sorted, cal_true_scores, side="left")
    pos_global = np.clip(pos_global, 0, max(global_sorted.size - 1, 0))
    r_global_loo_by_pos = _loo_rank_by_pos(global_sorted.size)
    Rg_loo = r_global_loo_by_pos[pos_global] if global_sorted.size > 0 else np.ones_like(cal_true_scores)

    # cluster LOO proxy: positions within clu_sorted[c]
    pos_in_cluster = np.full(Ncal, -1, dtype=np.int64)
    r_clu_loo_by_pos: List[np.ndarray] = []
    for c in range(Kc):
        nc = clu_sorted[c].size
        r_clu_loo_by_pos.append(_loo_rank_by_pos(nc))
        if nc == 0:
            continue
        idx = np.where(clu_of_point == c)[0]
        if idx.size == 0:
            continue
        vals = cal_true_scores[idx]
        pos = np.searchsorted(clu_sorted[c], vals, side="left")
        pos = np.clip(pos, 0, nc - 1)
        pos_in_cluster[idx] = pos.astype(np.int64)

    # build T_cal_by_cluster
    for c in range(Kc):
        idx = np.where(clu_of_point == c)[0]
        if idx.size == 0:
            continue
        nc = int(n_c[c])
        if nc <= 0:
            continue

        posc = np.clip(pos_in_cluster[idx], 0, max(r_clu_loo_by_pos[c].size - 1, 0))
        Rc_loo = r_clu_loo_by_pos[c][posc] if r_clu_loo_by_pos[c].size > 0 else np.ones(idx.size)
        Rg_vals = Rg_loo[idx]

        lam = float(tau / (tau + nc))
        T_cal_by_cluster[c] = ((1 - lam) * Rc_loo + lam * Rg_vals).astype(np.float64)

    covered = np.zeros(N, dtype=bool)
    set_sizes = np.zeros(N, dtype=np.int64)
    b_cache_local: Dict[int, np.ndarray] = {}

    for j in range(N):
        cand = cand_list[j]
        y_true = int(y_test[j])

        s_xy = softmax_score(p_test[j, cand]).astype(np.float64)

        Rg = _rank_from_sorted(global_sorted, s_xy)

        Rc = np.empty_like(Rg)
        nc_vec = np.empty_like(Rg, dtype=np.int64)

        for k_idx, y in enumerate(cand):
            y = int(y)
            c = int(cluster_id[y])
            if c == null_id or c < 0 or c >= Kc:
                # null cluster: no cluster info
                Rc[k_idx] = 1.0
                nc_vec[k_idx] = 0
            else:
                Rc[k_idx] = _rank_from_sorted(clu_sorted[c], np.array([s_xy[k_idx]]))[0]
                nc_vec[k_idx] = int(n_c[c])

        lam_vec = tau / (tau + np.maximum(nc_vec, 1))
        T_test = (1 - lam_vec) * Rc + lam_vec * Rg

        # re-conformalize within cluster
        u = np.empty_like(T_test, dtype=np.float64)
        for k_idx, y in enumerate(cand):
            y = int(y)
            c = int(cluster_id[y])
            if c == null_id or c < 0 or c >= Kc or n_c[c] <= 0:
                u[k_idx] = Rg[k_idx]  # fallback
                continue
            T_cal = T_cal_by_cluster[c]
            ge = int(np.sum(T_cal >= T_test[k_idx]))
            u[k_idx] = (1.0 + ge) / (float(n_c[c]) + 1.0)

        # optional CCV correction using cluster size n_c
        u_ccv = u.copy()
        if ccv_cache is not None:
            for k_idx, y in enumerate(cand):
                y = int(y)
                c = int(cluster_id[y])
                if c == null_id or c < 0 or c >= Kc:
                    continue
                nc = int(n_c[c])
                if nc <= 1 or nc > ccv_nmax:
                    continue

                if nc not in b_cache_local:
                    b = ccv_cache.get_b(nc)
                    if b is None:
                        b, gamma = precompute_b_table_gamma(nc, ccv_delta, ccv_cache.mc_B, ccv_cache.seed)
                        ccv_cache.set_b(nc, b, gamma)
                    b_cache_local[nc] = b
                b = b_cache_local[nc]
                u_ccv[k_idx] = ccv_h_from_b(b, np.array([u[k_idx]]))[0]

        keep = u_ccv > alpha
        set_sizes[j] = int(np.sum(keep))
        true_pos = int(np.where(cand == y_true)[0][0])
        covered[j] = bool(keep[true_pos])

    return covered, set_sizes


# -----------------------------
# Metrics
# -----------------------------
def tail_head_labels(n_y: np.ndarray, tail_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    K = n_y.size
    order = np.argsort(n_y)  # ascending
    m = int(np.floor(tail_frac * K))
    tail = order[:m]
    head = order[m:]
    return tail, head


def summarize(
    covered: np.ndarray,
    set_sizes: np.ndarray,
    y_test: np.ndarray,
    n_y_cal: np.ndarray,
    tail: np.ndarray,
    head: np.ndarray,
) -> Dict:
    out: Dict = {}
    out["coverage_marginal"] = float(np.mean(covered))
    out["size_mean"] = float(np.mean(set_sizes))
    out["size_median"] = float(np.median(set_sizes))

    tail_mask = np.isin(y_test, tail)
    head_mask = np.isin(y_test, head)

    out["coverage_tail"] = float(np.mean(covered[tail_mask])) if tail_mask.any() else float("nan")
    out["coverage_head"] = float(np.mean(covered[head_mask])) if head_mask.any() else float("nan")
    out["size_tail"] = float(np.mean(set_sizes[tail_mask])) if tail_mask.any() else float("nan")
    out["size_head"] = float(np.mean(set_sizes[head_mask])) if head_mask.any() else float("nan")

    K = n_y_cal.size
    cov_by_label = np.full(K, np.nan, dtype=np.float64)
    size_by_label = np.full(K, np.nan, dtype=np.float64)

    for y in range(K):
        idx = np.where(y_test == y)[0]
        if idx.size == 0:
            continue
        cov_by_label[y] = np.mean(covered[idx])
        size_by_label[y] = np.mean(set_sizes[idx])

    def _summ(arr, labels):
        v = arr[labels]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"mean": float("nan"), "q10": float("nan"), "q25": float("nan"), "worst": float("nan")}
        return {
            "mean": float(np.mean(v)),
            "q10": float(np.quantile(v, 0.10)),
            "q25": float(np.quantile(v, 0.25)),
            "worst": float(np.min(v)),
        }

    out["cov_label_all"] = _summ(cov_by_label, np.arange(K))
    out["cov_label_tail"] = _summ(cov_by_label, tail)
    out["cov_label_head"] = _summ(cov_by_label, head)

    out["cov_by_label"] = cov_by_label.tolist()
    out["size_by_label"] = size_by_label.tolist()
    return out


# -----------------------------
# Main experiment runner
# -----------------------------
def run_once(args, seed: int) -> Dict:
    d = load_npz(args.npz)
    check_npz(d)

    p_cal = d["p_cal"].astype(np.float64)
    y_cal = d["y_cal"].astype(np.int64)
    p_test = d["p_test"].astype(np.float64)
    y_test = d["y_test"].astype(np.int64)

    Ncal, K = p_cal.shape

    # candidates
    cand_list = make_candidates(p_test, y_test, args.topL)

    # tail/head labels
    n_y = np.bincount(y_cal, minlength=K).astype(np.int64)
    tail, head = tail_head_labels(n_y, args.tail_frac)

    # label clustering
    scores_cal = softmax_score(p_cal[np.arange(Ncal), y_cal]).astype(np.float64)

    E = build_label_embeddings_score_quantile(
        scores_cal=scores_cal,
        y_cal=y_cal,
        K=K,
        alpha=args.alpha,
        min_count=args.min_count,
        null_value=np.nan,
        q_grid=None,
    )

    cluster_id = cluster_labels_kmeans_cccp(
        E=E,
        counts=n_y,
        Kc=args.Kc,
        seed=seed,
        weighted=args.weighted_kmeans,
        min_count=args.min_count,
        null_id=args.null_id,
    )

    # CCV cache
    ccv_cache = None
    if args.use_ccv:
        ccv_cache = CCVCache(
            cache_dir=args.ccv_cache,
            delta=args.ccv_delta,
            mc_B=args.ccv_mc,
            seed=args.ccv_seed,
        )

    results = {
        "seed": seed,
        "alpha": args.alpha,
        "K": K,
        "topL": args.topL,
        "tail_frac": args.tail_frac,
        "Kc": args.Kc,
        "tau": args.tau,
        "min_count": args.min_count,
        "use_ccv": args.use_ccv,
    }

    cov_g, sz_g = eval_gcp(p_cal, y_cal, p_test, y_test, args.alpha, cand_list)
    results["gcp"] = summarize(cov_g, sz_g, y_test, n_y, tail, head)

    cov_o, sz_o = eval_ours_shrinkage_global_cluster_ccv(
        p_cal=p_cal,
        y_cal=y_cal,
        p_test=p_test,
        y_test=y_test,
        alpha=args.alpha,
        cand_list=cand_list,
        cluster_id=cluster_id,
        tau=args.tau,
        ccv_cache=ccv_cache,
        ccv_delta=args.ccv_delta,
        ccv_nmax=args.ccv_nmax,
        null_id=args.null_id,
    )
    results["ours_shrink_gc_ccv"] = summarize(cov_o, sz_o, y_test, n_y, tail, head)

    results["config"] = {
        "npz": args.npz,
        "score": "nonconformity=1-p_true",
        "Kc": args.Kc,
        "tau": args.tau,
        "min_count": args.min_count,
        "weighted_kmeans": args.weighted_kmeans,
        "null_id": args.null_id,
        "use_ccv": args.use_ccv,
        "ccv_delta": args.ccv_delta,
        "ccv_nmax": args.ccv_nmax,
        "ccv_mc": args.ccv_mc,
        "ccv_seed": args.ccv_seed,
        "topL": args.topL,
        "tail_frac": args.tail_frac,
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])

    # efficiency
    ap.add_argument("--topL", type=int, default=512, help="candidate labels per test; <=0 means all K")

    # tail definition
    ap.add_argument("--tail_frac", type=float, default=0.2)

    # clustering / ours
    ap.add_argument("--Kc", type=int, default=10)
    ap.add_argument("--tau", type=float, default=50.0)
    ap.add_argument("--weighted_kmeans", action="store_true")
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--null_id", type=int, default=-1)

    # CCV
    ap.add_argument("--use_ccv", action="store_true")
    ap.add_argument("--ccv_delta", type=float, default=0.1)
    ap.add_argument("--ccv_nmax", type=int, default=300, help="apply CCV only if cluster size <= this")
    ap.add_argument("--ccv_mc", type=int, default=30000)
    ap.add_argument("--ccv_seed", type=int, default=123)

    # output
    ap.add_argument("--out_dir", type=str, default="out/results")
    ap.add_argument("--ccv_cache", type=str, default="out/ccv_cache")

    args = ap.parse_args()

    cfg = {
        "alpha": args.alpha,
        "topL": args.topL,
        "tail_frac": args.tail_frac,
        "Kc": args.Kc,
        "tau": args.tau,
        "min_count": args.min_count,
        "weighted_kmeans": args.weighted_kmeans,
        "null_id": args.null_id,
        "use_ccv": args.use_ccv,
        "ccv_delta": args.ccv_delta,
        "ccv_nmax": args.ccv_nmax,
        "ccv_mc": args.ccv_mc,
        "ccv_seed": args.ccv_seed,
        "npz": os.path.basename(args.npz),
    }
    exp_id = f"inat_shrinkgc_{sha10(cfg)}"
    out_root = os.path.join(args.out_dir, exp_id)
    ensure_dir(out_root)

    all_rows = []
    for s in args.seeds:
        res = run_once(args, seed=s)

        with open(os.path.join(out_root, f"metrics_seed{s}.json"), "w") as f:
            json.dump(res, f, indent=2)

        for method in ["gcp", "ours_shrink_gc_ccv"]:
            m = res[method]
            row = {
                "seed": s,
                "method": method,
                "alpha": args.alpha,
                "topL": args.topL,
                "tail_frac": args.tail_frac,
                "Kc": args.Kc,
                "tau": args.tau,
                "min_count": args.min_count,
                "use_ccv": args.use_ccv,
                "coverage_marginal": m["coverage_marginal"],
                "size_mean": m["size_mean"],
                "coverage_tail": m["coverage_tail"],
                "coverage_head": m["coverage_head"],
                "size_tail": m["size_tail"],
                "size_head": m["size_head"],
                "cov_label_mean": m["cov_label_all"]["mean"],
                "cov_label_q10": m["cov_label_all"]["q10"],
                "cov_label_worst": m["cov_label_all"]["worst"],
                "cov_tail_mean": m["cov_label_tail"]["mean"],
                "cov_tail_q10": m["cov_label_tail"]["q10"],
                "cov_tail_worst": m["cov_label_tail"]["worst"],
            }
            all_rows.append(row)

    if len(all_rows) > 0:
        import csv
        csv_path = os.path.join(out_root, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"[DONE] exp_id={exp_id}")
    print(f"[OUT]  {out_root}")


if __name__ == "__main__":
    main()
