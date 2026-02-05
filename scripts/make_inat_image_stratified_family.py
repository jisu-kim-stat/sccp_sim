#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make iNat2017 image NPZ with FAMILY-level labels using TFDS TFRecord files directly (no tensorflow_datasets).

- Reads TFRecord shards from TFDS cache directory.
- Converts species label -> family label using category_to_family.json
- Builds stratified splits by FAMILY on a labeled pool (train+validation by default).
- Optionally uses TFDS test split TFRecords for the test set (recommended).

Output NPZ contains:
  X_train, y_train, X_calib, y_calib, X_test, y_test, meta
and optionally: idx_train, idx_calib, idx_test, family_ids, family_counts_*

IMPORTANT:
  224x224 uint8 images: 150,528 bytes each.
  If you save 50k+30k+10k = 90k images, raw array size ~ 13.5 GB RAM.
"""

import argparse
import json
import os
import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm


# -----------------------------
# TFRecord reading helpers
# -----------------------------
def _list_tfrecords(glob_pat: str) -> List[str]:
    files = sorted(glob.glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"No TFRecord files matched glob: {glob_pat}")
    return files


def _parse_example(serialized: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse TFDS example for i_naturalist2017.

    Usually contains:
      - 'image' : bytes
      - 'label' : int64
    We parse those two keys.
    """
    feat = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    ex = tf.io.parse_single_example(serialized, feat)
    return ex["image"], ex["label"]


def _decode_resize_uint8(image_bytes: tf.Tensor, image_size: int) -> tf.Tensor:
    """
    bytes -> HWC uint8 -> resize -> uint8 HWC
    """
    img = tf.io.decode_jpeg(image_bytes, channels=3)  # HWC uint8
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    img = tf.clip_by_value(img, 0, 255)
    img = tf.cast(img, tf.uint8)
    return img


def make_dataset_from_files(files: List[str], deterministic: bool = True) -> tf.data.Dataset:
    """
    Deterministic order: files sorted, and records in file order.
    """
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=1)
    opt = tf.data.Options()
    opt.experimental_deterministic = deterministic
    ds = ds.with_options(opt)
    return ds


def scan_family_labels(
    files: List[str],
    species_to_family: np.ndarray,
    min_family_count: int,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    First pass: scan only labels -> map to family -> return:
      - y_family_all (int64) length N_pool, with -1 for dropped/unmapped/rare families
      - keep_mask (bool) length N_pool
      - family_id_map (array of kept family ids, sorted)
    """
    ds = make_dataset_from_files(files, deterministic=True).map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    yfam = []
    for _, lab in tqdm(ds, desc=desc):
        sp = int(lab.numpy())
        fam = int(species_to_family[sp]) if 0 <= sp < species_to_family.size else -1
        yfam.append(fam)
    yfam = np.asarray(yfam, dtype=np.int64)

    # count families, exclude -1
    fam_vals, fam_counts = np.unique(yfam[yfam >= 0], return_counts=True)
    keep_fams = fam_vals[fam_counts >= min_family_count]
    keep_fams = np.asarray(sorted(keep_fams.tolist()), dtype=np.int64)

    keep_mask = np.isin(yfam, keep_fams)
    # remap kept family ids to 0..F-1 for training convenience
    fam_to_new = {int(f): i for i, f in enumerate(keep_fams.tolist())}
    y_new = np.full_like(yfam, -1, dtype=np.int64)
    idx_keep = np.where(keep_mask)[0]
    for i in idx_keep:
        y_new[i] = fam_to_new[int(yfam[i])]

    return y_new, keep_mask, keep_fams


# -----------------------------
# Stratified split (family-level)
# -----------------------------
def allocate_counts_proportional(class_sizes, total_target, rng: np.random.Generator) -> np.ndarray:
    """
    Pure proportional allocation with capacity constraint.
    """
    sizes = np.asarray(class_sizes, dtype=np.int64)
    K = sizes.size
    if total_target > int(sizes.sum()):
        raise ValueError(f"total_target={total_target} > available={int(sizes.sum())}")
    counts = np.zeros(K, dtype=np.int64)
    remaining = int(total_target)

    cap = sizes.copy()
    w = cap.astype(np.float64)
    wsum = float(w.sum())
    if wsum <= 0:
        idx = np.where(cap > 0)[0]
        rng.shuffle(idx)
        for k in idx:
            if remaining == 0:
                break
            take = min(int(cap[k]), remaining)
            counts[k] += take
            cap[k] -= take
            remaining -= take
        return counts

    raw = remaining * (w / wsum)
    add_floor = np.floor(raw).astype(np.int64)
    add = np.minimum(add_floor, cap)
    counts += add
    remaining -= int(add.sum())
    cap -= add

    if remaining == 0:
        return counts

    frac = raw - np.floor(raw)
    jitter = rng.random(K) * 1e-12
    order = np.argsort(-(frac + jitter))
    for k in order:
        if remaining == 0:
            break
        if cap[k] > 0:
            counts[k] += 1
            cap[k] -= 1
            remaining -= 1

    if remaining > 0:
        idx = np.where(cap > 0)[0]
        rng.shuffle(idx)
        for k in idx:
            if remaining == 0:
                break
            take = min(int(cap[k]), remaining)
            counts[k] += take
            cap[k] -= take
            remaining -= take

    return counts


def stratified_split_indices_from_family_labels(
    y_family: np.ndarray,
    n_train: int,
    n_calib: int,
    n_test: int,
    seed: int,
    min_test_per_family: int = 1,
    min_calib_per_family: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    y_family: length N_pool, values in {0..F-1}, and assumes all are valid (no -1).
    Returns:
      idx_train, idx_calib, idx_test (pool indices), and family_counts (F,)
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_family, dtype=np.int64)
    F = int(y.max()) + 1

    family_counts = np.bincount(y, minlength=F).astype(np.int64)
    total = int(y.size)
    target = n_train + n_calib + n_test
    if target > total:
        raise ValueError(f"Requested {target} samples but only {total} available (after filtering).")

    # per-family index lists
    idx_by_f = [np.where(y == f)[0] for f in range(F)]
    for f in range(F):
        rng.shuffle(idx_by_f[f])

    # base mins feasibility: require enough to allocate mins + leave at least 1 for train
    feasible = family_counts >= (min_test_per_family + min_calib_per_family + 1)

    # allocate test
    test_counts = allocate_counts_proportional(family_counts, n_test, rng=rng)
    if min_test_per_family > 0:
        for f in np.where(feasible)[0]:
            if test_counts[f] < min_test_per_family:
                need = min_test_per_family - int(test_counts[f])
                donors = np.where(test_counts > min_test_per_family)[0]
                for d in donors:
                    take = min(int(test_counts[d] - min_test_per_family), need)
                    if take > 0:
                        test_counts[d] -= take
                        test_counts[f] += take
                        need -= take
                    if need == 0:
                        break

    # allocate calib from remaining
    remaining_after_test = family_counts - test_counts
    cap_for_calib = remaining_after_test.copy()
    cap_for_calib[feasible] = np.maximum(0, cap_for_calib[feasible] - 1)  # leave >=1 for train

    calib_counts = np.zeros(F, dtype=np.int64)
    if min_calib_per_family > 0:
        for f in np.where(feasible)[0]:
            if cap_for_calib[f] >= min_calib_per_family:
                calib_counts[f] = min_calib_per_family

    remaining_calib = n_calib - int(calib_counts.sum())
    if remaining_calib < 0:
        calib_counts[:] = 0
        remaining_calib = n_calib

    if remaining_calib > 0:
        cap_left = cap_for_calib - calib_counts
        add = allocate_counts_proportional(cap_left, remaining_calib, rng=rng)
        calib_counts += add

    # train = leftover, then subsample to n_train if needed
    train_counts = family_counts - test_counts - calib_counts
    if int(train_counts.sum()) < n_train:
        raise ValueError("Not enough remaining samples for train after test+calib allocation.")
    if int(train_counts.sum()) > n_train:
        train_counts = allocate_counts_proportional(train_counts, n_train, rng=rng)

    idx_train, idx_calib, idx_test = [], [], []
    for f in range(F):
        idx_f = idx_by_f[f]
        nt = int(test_counts[f])
        nc = int(calib_counts[f])
        nr = int(train_counts[f])
        need = nt + nc + nr
        if need > idx_f.size:
            # trim train first
            overflow = need - idx_f.size
            nr = max(0, nr - overflow)

        idx_test.extend(idx_f[:nt])
        idx_calib.extend(idx_f[nt:nt + nc])
        idx_train.extend(idx_f[nt + nc:nt + nc + nr])

    idx_train = np.asarray(idx_train, dtype=np.int64)
    idx_calib = np.asarray(idx_calib, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)

    rng.shuffle(idx_train)
    rng.shuffle(idx_calib)
    rng.shuffle(idx_test)

    # disjoint checks
    assert np.intersect1d(idx_train, idx_calib).size == 0
    assert np.intersect1d(idx_train, idx_test).size == 0
    assert np.intersect1d(idx_calib, idx_test).size == 0

    return idx_train, idx_calib, idx_test, family_counts


def collect_images_by_pool_indices(
    files: List[str],
    pool_indices_set: set,
    n_take: int,
    y_family_full: np.ndarray,
    image_size: int,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Second pass: iterate TFRecords in deterministic order; collect only indices in pool_indices_set.
    y_family_full: length N_pool, mapped family labels (0..F-1), for valid pool positions.
    """
    ds = make_dataset_from_files(files, deterministic=True).map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    X = np.empty((n_take, 3, image_size, image_size), dtype=np.uint8)
    y = np.empty((n_take,), dtype=np.int64)

    j = 0
    for i, (img_b, _) in enumerate(tqdm(ds, desc=desc)):
        if i not in pool_indices_set:
            continue
        img = _decode_resize_uint8(img_b, image_size)  # HWC uint8
        x = img.numpy()
        X[j] = np.transpose(x, (2, 0, 1))  # CHW
        y[j] = int(y_family_full[i])
        j += 1
        if j == n_take:
            break

    if j != n_take:
        raise RuntimeError(f"Collected {j}/{n_take} samples for {desc}. Pool indexing/order mismatch.")
    return X, y


# -----------------------------
# Species -> Family mapping
# -----------------------------
def load_species_to_family(cat_to_family_json: str, K_species: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    category_to_family.json: { "1916": {"family_id": ..., "family_name": ...}, ... }
    Returns:
      - species_to_family_id: shape (K_species,), int64, -1 if missing
      - unique_family_ids_sorted: all family_ids present (excluding None)
    """
    with open(cat_to_family_json, "r", encoding="utf-8") as f:
        m = json.load(f)

    species_to_family = np.full((K_species,), -1, dtype=np.int64)
    fam_ids = []
    for k_str, v in m.items():
        try:
            sp = int(k_str)
        except Exception:
            continue
        if sp < 0 or sp >= K_species:
            continue
        fam_id = v.get("family_id", None)
        if fam_id is None:
            continue
        fam_id = int(fam_id)
        species_to_family[sp] = fam_id
        fam_ids.append(fam_id)

    fam_ids = np.asarray(sorted(set(fam_ids)), dtype=np.int64)
    return species_to_family, fam_ids


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # TFRecord location (TFDS cache)
    ap.add_argument("--tfds_dir", type=str, required=True,
                    help="TFDS i_naturalist2017 version dir, e.g. data/tfds/i_naturalist2017/0.1.0")

    ap.add_argument("--train_glob", type=str, default="i_naturalist2017-train.tfrecord-*-of-*",
                    help="Glob (relative to --tfds_dir) for train TFRecord shards")
    ap.add_argument("--val_glob", type=str, default="i_naturalist2017-validation.tfrecord-*-of-*",
                    help="Glob (relative to --tfds_dir) for validation TFRecord shards (optional; if none found, ignored)")
    ap.add_argument("--test_glob", type=str, default="i_naturalist2017-test.tfrecord-*-of-*",
                    help="Glob (relative to --tfds_dir) for test TFRecord shards")

    # taxonomy mapping
    ap.add_argument("--cat_to_family_json", type=str, required=True,
                    help="category_to_family.json path (species id -> family id)")

    # output
    ap.add_argument("--out_npz", type=str, required=True)

    # sampling
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_calib", type=int, default=30000)
    ap.add_argument("--n_test", type=int, default=10000)

    # family filtering / stratify
    ap.add_argument("--min_family_count", type=int, default=250,
                    help="keep only families with at least this many samples in the POOL (train+val)")
    ap.add_argument("--min_test_per_family", type=int, default=1)
    ap.add_argument("--min_calib_per_family", type=int, default=1)

    ap.add_argument("--use_tfds_test_for_test", action="store_true",
                    help="If set: test split is drawn from TFDS test TFRecords (recommended). "
                         "If not set: test is also drawn from pool (train+val) via stratification.")

    ap.add_argument("--save_indices", action="store_true",
                    help="Also save pool indices for each split and family count summaries.")

    args = ap.parse_args()

    tfds_dir = os.path.expanduser(args.tfds_dir)
    out_npz = os.path.expanduser(args.out_npz)
    cat_to_family_json = os.path.expanduser(args.cat_to_family_json)

    # --- resolve tfrecords ---
    train_files = _list_tfrecords(os.path.join(tfds_dir, args.train_glob))

    val_pat = os.path.join(tfds_dir, args.val_glob)
    val_files = sorted(glob.glob(val_pat))
    if not val_files:
        print(f"[WARN] no validation TFRecords matched {val_pat}. Pool will be train-only.")
        pool_files = train_files
        has_val = False
    else:
        pool_files = train_files + sorted(val_files)
        has_val = True

    test_files = _list_tfrecords(os.path.join(tfds_dir, args.test_glob))

    print(f"[tfrecord] train shards: {len(train_files)}")
    print(f"[tfrecord] val shards  : {len(val_files)} (used={has_val})")
    print(f"[tfrecord] test shards : {len(test_files)}")
    print(f"[pool] shards total   : {len(pool_files)}")

    # --- load mapping ---
    # Species K is known to be 5089 for iNat2017, but we infer as max key + 1 if possible.
    # For safety, set K_species=5089 if you know it; otherwise we can take a conservative bound.
    K_species = 5089
    species_to_family, _ = load_species_to_family(cat_to_family_json, K_species=K_species)

    # --- pass1: scan pool family labels + filter families by min_family_count ---
    y_pool_family_new, keep_mask, keep_family_ids = scan_family_labels(
        files=pool_files,
        species_to_family=species_to_family,
        min_family_count=args.min_family_count,
        desc="[pass1] scan pool labels -> family",
    )
    pool_idx_keep = np.where(keep_mask)[0]
    print(f"[pool] total examples: {keep_mask.size}")
    print(f"[pool] kept examples : {pool_idx_keep.size} (after min_family_count={args.min_family_count})")
    print(f"[pool] kept families : {keep_family_ids.size}")

    # restrict to kept indices
    y_pool_kept = y_pool_family_new[pool_idx_keep]
    if np.any(y_pool_kept < 0):
        raise RuntimeError("Internal error: kept pool contains -1 labels.")

    # --- split indices ---
    rng = np.random.default_rng(args.seed)

    if args.use_tfds_test_for_test:
        # train/calib drawn from pool only; test drawn from TFDS test split
        # We still stratify train/calib by family on pool.
        n_pool_needed = args.n_train + args.n_calib
        if n_pool_needed > y_pool_kept.size:
            raise ValueError(f"Need {n_pool_needed} pool samples but only {y_pool_kept.size} available after filtering.")

        # To stratify train/calib only, we do a pseudo split with n_test=0 then ignore test.
        idx_tr_rel, idx_cal_rel, idx_te_rel, fam_counts = stratified_split_indices_from_family_labels(
            y_family=y_pool_kept,
            n_train=args.n_train,
            n_calib=args.n_calib,
            n_test=0,
            seed=args.seed,
            min_test_per_family=0,
            min_calib_per_family=args.min_calib_per_family,
        )
        # map relative indices (within kept pool) back to absolute pool indices
        idx_train = pool_idx_keep[idx_tr_rel]
        idx_calib = pool_idx_keep[idx_cal_rel]

        # --- build TFDS test family labels (filter to same family set) ---
        # scan test, map to same kept families (by family_id -> new_id mapping)
        # Build family_id -> new_id
        fam_to_new = {int(fam_id): i for i, fam_id in enumerate(keep_family_ids.tolist())}

        ds_test = make_dataset_from_files(test_files, deterministic=True).map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        y_test_new = []
        keep_test_mask = []
        for _, lab in tqdm(ds_test, desc="[pass1b] scan TFDS test labels -> family"):
            sp = int(lab.numpy())
            fam_id = int(species_to_family[sp]) if 0 <= sp < species_to_family.size else -1
            new_id = fam_to_new.get(fam_id, None)
            if new_id is None:
                y_test_new.append(-1)
                keep_test_mask.append(False)
            else:
                y_test_new.append(int(new_id))
                keep_test_mask.append(True)
        y_test_new = np.asarray(y_test_new, dtype=np.int64)
        keep_test_mask = np.asarray(keep_test_mask, dtype=bool)
        test_keep_idx = np.where(keep_test_mask)[0]
        if test_keep_idx.size < args.n_test:
            raise ValueError(
                f"TFDS test has only {test_keep_idx.size} samples in kept families, but n_test={args.n_test} requested. "
                f"Lower n_test or lower min_family_count."
            )
        # sample n_test uniformly from available test_keep_idx
        rng.shuffle(test_keep_idx)
        idx_test = test_keep_idx[:args.n_test]
        y_test_full = y_test_new  # for collecting we need label by absolute test index
        print(f"[test] TFDS test total={keep_test_mask.size}, kept={test_keep_idx.size}, taking={args.n_test}")

    else:
        # train/calib/test all from pool (train+val) after filtering
        idx_tr_rel, idx_cal_rel, idx_te_rel, fam_counts = stratified_split_indices_from_family_labels(
            y_family=y_pool_kept,
            n_train=args.n_train,
            n_calib=args.n_calib,
            n_test=args.n_test,
            seed=args.seed,
            min_test_per_family=args.min_test_per_family,
            min_calib_per_family=args.min_calib_per_family,
        )
        idx_train = pool_idx_keep[idx_tr_rel]
        idx_calib = pool_idx_keep[idx_cal_rel]
        idx_test = pool_idx_keep[idx_te_rel]
        y_test_full = None  # not used in this branch

    print("[split sizes]",
          "train", idx_train.size,
          "calib", idx_calib.size,
          "test", idx_test.size)

    # --- pass2: collect images ---
    set_tr = set(map(int, idx_train.tolist()))
    set_cal = set(map(int, idx_calib.tolist()))

    # We need full y_pool_family_new to label by absolute pool index
    # But y_pool_family_new has -1 on dropped indices; that's fine since we only collect from kept indices.
    X_train, y_train = collect_images_by_pool_indices(
        files=pool_files,
        pool_indices_set=set_tr,
        n_take=args.n_train,
        y_family_full=y_pool_family_new,
        image_size=args.image_size,
        desc="[pass2] collect train",
    )

    X_calib, y_calib = collect_images_by_pool_indices(
        files=pool_files,
        pool_indices_set=set_cal,
        n_take=args.n_calib,
        y_family_full=y_pool_family_new,
        image_size=args.image_size,
        desc="[pass2] collect calib",
    )

    if args.use_tfds_test_for_test:
        set_te = set(map(int, idx_test.tolist()))
        # collect from TFDS test files, with y_test_full giving labels by test absolute index
        ds = make_dataset_from_files(test_files, deterministic=True).map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)

        X_test = np.empty((args.n_test, 3, args.image_size, args.image_size), dtype=np.uint8)
        y_test = np.empty((args.n_test,), dtype=np.int64)
        j = 0
        for i, (img_b, _) in enumerate(tqdm(ds, desc="[pass2] collect TFDS test")):
            if i not in set_te:
                continue
            img = _decode_resize_uint8(img_b, args.image_size)
            x = img.numpy()
            X_test[j] = np.transpose(x, (2, 0, 1))
            y_test[j] = int(y_test_full[i])
            j += 1
            if j == args.n_test:
                break
        if j != args.n_test:
            raise RuntimeError(f"Collected {j}/{args.n_test} test samples. Order mismatch?")
    else:
        set_te = set(map(int, idx_test.tolist()))
        X_test, y_test = collect_images_by_pool_indices(
            files=pool_files,
            pool_indices_set=set_te,
            n_take=args.n_test,
            y_family_full=y_pool_family_new,
            image_size=args.image_size,
            desc="[pass2] collect test",
        )

    # --- meta + save ---
    meta = {
        "tfds_dir": tfds_dir,
        "train_glob": args.train_glob,
        "val_glob": args.val_glob,
        "test_glob": args.test_glob,
        "pool_uses_val": bool(has_val),
        "cat_to_family_json": cat_to_family_json,
        "seed": int(args.seed),
        "image_size": int(args.image_size),
        "format": "uint8_CHW",
        "n_train": int(args.n_train),
        "n_calib": int(args.n_calib),
        "n_test": int(args.n_test),
        "min_family_count": int(args.min_family_count),
        "min_test_per_family": int(args.min_test_per_family),
        "min_calib_per_family": int(args.min_calib_per_family),
        "use_tfds_test_for_test": bool(args.use_tfds_test_for_test),
        "note": "Family-level stratified sampling from TFDS TFRecords (no tfds dependency).",
        "kept_family_ids": keep_family_ids.tolist(),  # original iNat family taxon ids
    }

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    save_kwargs = dict(
        X_train=X_train, y_train=y_train,
        X_calib=X_calib, y_calib=y_calib,
        X_test=X_test,   y_test=y_test,
        meta=np.array([meta], dtype=object),
    )

    if args.save_indices:
        save_kwargs.update(
            idx_train=idx_train.astype(np.int64),
            idx_calib=idx_calib.astype(np.int64),
            idx_test=idx_test.astype(np.int64),
            kept_pool_indices=pool_idx_keep.astype(np.int64),
        )

    np.savez_compressed(out_npz, **save_kwargs)
    print(f"[saved] {out_npz}")
    print("[shapes]",
          "X_train", X_train.shape, "y_train", y_train.shape,
          "X_calib", X_calib.shape, "y_calib", y_calib.shape,
          "X_test",  X_test.shape,  "y_test",  y_test.shape)


if __name__ == "__main__":
    main()
