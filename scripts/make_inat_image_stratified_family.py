#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(inat_root: str, file_name: str) -> str:
    """
    iNat json의 images[].file_name은 환경마다
    - "train_val2017/xxxx.jpg"
    - "xxxx.jpg"
    - "train_val2017/train_val2017/xxxx.jpg"
    등으로 다양할 수 있어서 몇 가지 후보를 순서대로 확인.
    """
    candidates = [
        os.path.join(inat_root, file_name),
        os.path.join(inat_root, "train_val2017", file_name),
        os.path.join(inat_root, "train_val2017", "train_val2017", file_name),
        os.path.join(inat_root, "train2017", file_name),
        os.path.join(inat_root, "val2017", file_name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # 마지막: file_name의 basename만 붙여보는 fallback
    base = os.path.basename(file_name)
    candidates2 = [
        os.path.join(inat_root, base),
        os.path.join(inat_root, "train_val2017", base),
        os.path.join(inat_root, "train_val2017", "train_val2017", base),
        os.path.join(inat_root, "train2017", base),
        os.path.join(inat_root, "val2017", base),
    ]
    for p in candidates2:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not resolve image path for file_name={file_name}")


def load_and_resize_uint8_chw(img_path: str, image_size: int) -> np.ndarray:
    """
    return: uint8 CHW (3, H, W)
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize((image_size, image_size), resample=Image.BILINEAR)
        x = np.asarray(im, dtype=np.uint8)  # HWC
    return np.transpose(x, (2, 0, 1))      # CHW


# -----------------------------
# Stratified splitting utilities
# (너가 쓰던 로직 그대로/유사하게)
# -----------------------------
def allocate_counts_proportional(
    class_sizes: np.ndarray,
    total_target: int,
    min_per_class: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    integer counts per class ~ proportional to class_sizes
    with sum==total_target (best effort) and counts<=class_sizes
    """
    sizes = np.asarray(class_sizes, dtype=np.int64)
    K = sizes.size

    if total_target > int(sizes.sum()):
        raise ValueError(f"total_target={total_target} > total_available={int(sizes.sum())}")

    counts = np.zeros(K, dtype=np.int64)

    if min_per_class > 0:
        feasible = sizes >= min_per_class
        counts[feasible] = min_per_class
        remaining = total_target - int(counts.sum())
        if remaining < 0:
            counts[:] = 0
            remaining = total_target
    else:
        remaining = total_target

    cap = sizes - counts
    if remaining == 0:
        return counts

    w = sizes.astype(np.float64)
    wsum = float(w.sum())
    if wsum <= 0:
        idx = np.where(cap > 0)[0]
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(idx)
        for c in idx:
            take = min(int(cap[c]), remaining)
            counts[c] += take
            remaining -= take
            if remaining == 0:
                break
        return counts

    raw = remaining * (w / wsum)
    add_floor = np.floor(raw).astype(np.int64)
    add = np.minimum(add_floor, cap)
    counts += add
    remaining -= int(add.sum())
    cap = sizes - counts

    if remaining == 0:
        return counts

    frac = raw - np.floor(raw)
    order = np.argsort(-frac)
    if rng is not None:
        jitter = rng.random(K) * 1e-12
        order = np.argsort(-(frac + jitter))

    for c in order:
        if remaining == 0:
            break
        if cap[c] > 0:
            counts[c] += 1
            cap[c] -= 1
            remaining -= 1

    if remaining > 0:
        idx = np.where(cap > 0)[0]
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(idx)
        for c in idx:
            if remaining == 0:
                break
            take = min(int(cap[c]), remaining)
            counts[c] += take
            remaining -= take

    return counts


def stratified_split_indices(
    labels: np.ndarray,
    n_train: int,
    n_calib: int,
    n_test: int,
    seed: int = 1,
    min_test_per_class: int = 1,
    min_calib_per_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    labels: (N,) int in {0..F-1}
    returns:
      idx_train, idx_calib, idx_test,
      classes (0..F-1 actually), sizes,
      train_counts, calib_counts, test_counts
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)

    classes, sizes = np.unique(labels, return_counts=True)
    # assume classes are 0..F-1 contiguous, but still keep classes
    F = classes.size

    total = labels.size
    if n_train + n_calib + n_test > total:
        raise ValueError(f"Requested {n_train+n_calib+n_test} but only {total} available.")

    # feasibility for minimum constraints while leaving >=1 for train
    min_tc = min_test_per_class + min_calib_per_class + 1
    feasible_tc = sizes >= min_tc

    # test allocation
    test_counts = allocate_counts_proportional(sizes, n_test, min_per_class=0, rng=rng)
    if min_test_per_class > 0:
        for k in np.where(feasible_tc)[0]:
            if test_counts[k] < min_test_per_class:
                need = min_test_per_class - int(test_counts[k])
                donors = np.where(test_counts > min_test_per_class)[0]
                for d in donors:
                    take = min(int(test_counts[d] - min_test_per_class), need)
                    if take > 0:
                        test_counts[d] -= take
                        test_counts[k] += take
                        need -= take
                    if need == 0:
                        break

    # calib allocation from remaining
    rem_after_test = sizes - test_counts
    cap_cal = rem_after_test.copy()
    cap_cal[feasible_tc] = np.maximum(0, cap_cal[feasible_tc] - 1)  # leave >=1 for train

    calib_counts = np.zeros(F, dtype=np.int64)
    if min_calib_per_class > 0:
        for k in np.where(feasible_tc)[0]:
            if cap_cal[k] >= min_calib_per_class:
                calib_counts[k] = min_calib_per_class

    remaining_calib = n_calib - int(calib_counts.sum())
    if remaining_calib < 0:
        calib_counts[:] = 0
        remaining_calib = n_calib

    if remaining_calib > 0:
        cap_left = cap_cal - calib_counts
        weights = rem_after_test.astype(np.float64)
        wsum = float(weights.sum())
        if wsum <= 0:
            idx = np.where(cap_left > 0)[0]
            rng.shuffle(idx)
            for k in idx:
                if remaining_calib == 0:
                    break
                take = min(int(cap_left[k]), remaining_calib)
                calib_counts[k] += take
                remaining_calib -= take
        else:
            raw = remaining_calib * (weights / wsum)
            add_floor = np.floor(raw).astype(np.int64)
            add = np.minimum(add_floor, cap_left)
            calib_counts += add
            remaining_calib -= int(add.sum())
            cap_left = cap_cal - calib_counts

            if remaining_calib > 0:
                frac = raw - np.floor(raw)
                jitter = rng.random(F) * 1e-12
                order = np.argsort(-(frac + jitter))
                for k in order:
                    if remaining_calib == 0:
                        break
                    if cap_left[k] > 0:
                        calib_counts[k] += 1
                        cap_left[k] -= 1
                        remaining_calib -= 1

                if remaining_calib > 0:
                    idx = np.where(cap_left > 0)[0]
                    rng.shuffle(idx)
                    for k in idx:
                        if remaining_calib == 0:
                            break
                        take = min(int(cap_left[k]), remaining_calib)
                        calib_counts[k] += take
                        remaining_calib -= take

    # train allocation
    train_counts = sizes - test_counts - calib_counts
    if int(train_counts.sum()) < n_train:
        raise ValueError("Not enough remaining samples for train after test+calib allocation.")
    if int(train_counts.sum()) > n_train:
        train_counts = allocate_counts_proportional(train_counts, n_train, min_per_class=0, rng=rng)

    # pick indices
    idx_train, idx_cal, idx_test = [], [], []
    for k, c in enumerate(classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        nt = int(test_counts[k])
        nc = int(calib_counts[k])
        nr = int(train_counts[k])

        need = nt + nc + nr
        if need > idx_c.size:
            overflow = need - idx_c.size
            nr = max(0, nr - overflow)

        idx_test.extend(idx_c[:nt])
        idx_cal.extend(idx_c[nt:nt+nc])
        idx_train.extend(idx_c[nt+nc:nt+nc+nr])

    idx_train = np.asarray(idx_train, dtype=np.int64)
    idx_cal   = np.asarray(idx_cal, dtype=np.int64)
    idx_test  = np.asarray(idx_test, dtype=np.int64)

    rng.shuffle(idx_train)
    rng.shuffle(idx_cal)
    rng.shuffle(idx_test)

    assert np.intersect1d(idx_train, idx_cal).size == 0
    assert np.intersect1d(idx_train, idx_test).size == 0
    assert np.intersect1d(idx_cal, idx_test).size == 0

    return idx_train, idx_cal, idx_test, classes, sizes, train_counts, calib_counts, test_counts


# -----------------------------
# Build pool from train+val json
# -----------------------------
def build_pool_from_json(
    inat_root: str,
    train_json: str,
    val_json: str,
    cat_to_family_json: str,
    min_family_count: int = 0,
    drop_unmapped: bool = True,
) -> Tuple[List[str], np.ndarray, Dict]:
    """
    Returns:
      paths: list of image paths (len=Npool)
      yfam : np.ndarray family labels in 0..F-1 (len=Npool)
      meta : dict with mappings and counts
    """
    inat_root = os.path.expanduser(inat_root)
    train_json = os.path.expanduser(train_json)
    val_json   = os.path.expanduser(val_json)
    cat_to_family_json = os.path.expanduser(cat_to_family_json)

    cat_to_family = load_json(cat_to_family_json)  # keys are strings usually
    # category_id(str/int) -> family_id
    def map_cat_to_fam(cat_id: int) -> Optional[int]:
        v = cat_to_family.get(str(cat_id), None)
        if v is None:
            return None
        fid = v.get("family_id", None)
        if fid is None:
            return None
        return int(fid)

    def read_split(split_json_path: str):
        data = load_json(split_json_path)
        images = {int(x["id"]): x["file_name"] for x in data["images"]}
        anns = data["annotations"]
        return images, anns

    img_map_tr, ann_tr = read_split(train_json)
    img_map_va, ann_va = read_split(val_json)

    paths: List[str] = []
    fam_ids: List[int] = []
    cat_ids: List[int] = []

    # helper to add annotations
    def consume(img_map, anns):
        for a in anns:
            img_id = int(a["image_id"])
            cat_id = int(a["category_id"])
            fn = img_map.get(img_id, None)
            if fn is None:
                continue
            fid = map_cat_to_fam(cat_id)
            if fid is None:
                if drop_unmapped:
                    continue
                else:
                    # unmapped -> skip anyway (family classification 불가)
                    continue
            p = resolve_image_path(inat_root, fn)
            paths.append(p)
            fam_ids.append(fid)
            cat_ids.append(cat_id)

    consume(img_map_tr, ann_tr)
    consume(img_map_va, ann_va)

    fam_ids = np.asarray(fam_ids, dtype=np.int64)
    cat_ids = np.asarray(cat_ids, dtype=np.int64)

    # reindex family_id -> 0..F-1 (model 학습용)
    uniq_fam, inv = np.unique(fam_ids, return_inverse=True)
    yfam = inv.astype(np.int64)

    # optionally drop small families
    if min_family_count > 0:
        counts = np.bincount(yfam)
        keep_f = np.where(counts >= min_family_count)[0]
        keep_mask = np.isin(yfam, keep_f)
        paths = [p for p, m in zip(paths, keep_mask.tolist()) if m]
        yfam = yfam[keep_mask]

        # recompute reindex after filtering
        uniq_fam2, inv2 = np.unique(uniq_fam[keep_f], return_inverse=True)
        # map old yfam values (subset) -> new contiguous
        # easiest: rebuild from original family ids of kept samples
        fam_ids_kept = fam_ids[keep_mask]
        uniq_fam3, inv3 = np.unique(fam_ids_kept, return_inverse=True)
        uniq_fam = uniq_fam3
        yfam = inv3.astype(np.int64)
        counts = np.bincount(yfam)
    else:
        counts = np.bincount(yfam)

    meta = {
        "inat_root": inat_root,
        "train_json": train_json,
        "val_json": val_json,
        "cat_to_family_json": cat_to_family_json,
        "N_pool": int(len(paths)),
        "num_families": int(np.unique(yfam).size),
        "min_family_count_used": int(min_family_count),
        "family_id_values": uniq_fam.tolist(),  # new label i corresponds to family_id_values[i]
        "family_counts": counts.tolist(),
    }
    return paths, yfam, meta


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inat_root", type=str, required=True, help="extracted iNat root dir (contains train2017.json, val2017.json, and images folders)")
    ap.add_argument("--train_json", type=str, required=True, help="path to train2017.json")
    ap.add_argument("--val_json", type=str, required=True, help="path to val2017.json")
    ap.add_argument("--cat_to_family_json", type=str, required=True, help="category_to_family.json produced by your downloader")
    ap.add_argument("--out_npz", type=str, required=True)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=224)

    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_calib", type=int, default=30000)
    ap.add_argument("--n_test",  type=int, default=10000)

    ap.add_argument("--min_family_count", type=int, default=0,
                    help="drop families with < this count BEFORE sampling (recommended to avoid tons of tiny families)")
    ap.add_argument("--min_test_per_family", type=int, default=1)
    ap.add_argument("--min_calib_per_family", type=int, default=1)

    ap.add_argument("--save_indices", action="store_true",
                    help="store split indices and per-family counts in the NPZ")

    args = ap.parse_args()

    out_npz = os.path.expanduser(args.out_npz)
    ensure_dir(os.path.dirname(out_npz))

    # 1) build pool
    paths, yfam, pool_meta = build_pool_from_json(
        inat_root=args.inat_root,
        train_json=args.train_json,
        val_json=args.val_json,
        cat_to_family_json=args.cat_to_family_json,
        min_family_count=args.min_family_count,
        drop_unmapped=True,
    )

    Npool = len(paths)
    F = int(np.unique(yfam).size)
    print(f"[pool] N={Npool} | #families={F}")
    print(f"[pool] min/median/mean/max family count: "
          f"{int(np.bincount(yfam).min())} / {float(np.median(np.bincount(yfam)))} / {float(np.mean(np.bincount(yfam)))} / {int(np.bincount(yfam).max())}")

    target_total = args.n_train + args.n_calib + args.n_test
    if target_total > Npool:
        raise ValueError(f"Requested {target_total} but pool has only {Npool} samples.")

    # 2) stratified split (family 기준)
    idx_tr, idx_cal, idx_te, fam_values, fam_sizes, tr_ct, cal_ct, te_ct = stratified_split_indices(
        labels=yfam,
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_test=args.n_test,
        seed=args.seed,
        min_test_per_class=args.min_test_per_family,
        min_calib_per_class=args.min_calib_per_family,
    )
    print(f"[split] train={idx_tr.size} calib={idx_cal.size} test={idx_te.size}")

    # 3) load images and store
    def collect(idxs: np.ndarray, desc: str) -> Tuple[np.ndarray, np.ndarray]:
        X = np.empty((idxs.size, 3, args.image_size, args.image_size), dtype=np.uint8)
        y = np.empty((idxs.size,), dtype=np.int64)
        for j, i in enumerate(tqdm(idxs.tolist(), desc=desc)):
            X[j] = load_and_resize_uint8_chw(paths[i], args.image_size)
            y[j] = int(yfam[i])
        return X, y

    X_train, y_train = collect(idx_tr, "[load] train")
    X_calib, y_calib = collect(idx_cal, "[load] calib")
    X_test,  y_test  = collect(idx_te, "[load] test")

    meta = {
        **pool_meta,
        "seed": int(args.seed),
        "image_size": int(args.image_size),
        "format": "uint8_CHW",
        "n_train": int(args.n_train),
        "n_calib": int(args.n_calib),
        "n_test": int(args.n_test),
        "min_test_per_family": int(args.min_test_per_family),
        "min_calib_per_family": int(args.min_calib_per_family),
        "note": "Family-level stratified sampling from iNat train+val pool using taxonomy mapping (category->family).",
    }

    save_kwargs = dict(
        X_train=X_train, y_train=y_train,
        X_calib=X_calib, y_calib=y_calib,
        X_test=X_test,   y_test=y_test,
        meta=np.array([meta], dtype=object),
    )

    if args.save_indices:
        save_kwargs.update(
            idx_train=idx_tr.astype(np.int64),
            idx_calib=idx_cal.astype(np.int64),
            idx_test=idx_te.astype(np.int64),
            family_sizes=np.bincount(yfam).astype(np.int64),
            family_counts_train=tr_ct.astype(np.int64),
            family_counts_calib=cal_ct.astype(np.int64),
            family_counts_test=te_ct.astype(np.int64),
        )

    np.savez_compressed(out_npz, **save_kwargs)
    print(f"[saved] {out_npz}")
    print("[shapes]",
          "X_train", X_train.shape, "y_train", y_train.shape,
          "X_calib", X_calib.shape, "y_calib", y_calib.shape,
          "X_test",  X_test.shape,  "y_test",  y_test.shape)


if __name__ == "__main__":
    main()
