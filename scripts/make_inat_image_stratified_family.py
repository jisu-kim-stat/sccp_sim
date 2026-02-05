#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


# -----------------------------
# IO / small utils
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resize_uint8(img: tf.Tensor, size: int) -> tf.Tensor:
    """
    img: HWC uint8 (or convertible)
    returns: HWC uint8 resized to (size,size)
    """
    if img.dtype != tf.uint8:
        img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
    img = tf.image.resize(img, (size, size), method="bilinear")
    img = tf.clip_by_value(img, 0, 255)
    return tf.cast(img, tf.uint8)


def load_cat_to_family(cat_to_family_json: str) -> Dict[int, int]:
    """
    category_to_family.json:
      { "<species_id>": {"family_id": <int or null>, "family_name": <str or null>}, ... }
    return: species_id -> family_id (only if family_id exists)
    """
    with open(cat_to_family_json, "r", encoding="utf-8") as f:
        d = json.load(f)

    out = {}
    for k, v in d.items():
        try:
            sid = int(k)
        except Exception:
            continue
        fam = v.get("family_id", None) if isinstance(v, dict) else None
        if fam is None:
            continue
        try:
            out[sid] = int(fam)
        except Exception:
            continue
    return out


def build_split_codes_by_family(
    fam_raw: np.ndarray,
    min_family_count: int,
    seed: int,
    fracs: Tuple[float, float, float, float],  # train, sel, cal, test
    min_sel_per_family: int,
    min_cal_per_family: int,
    min_test_per_family: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    fam_raw: length N array, raw family_id (>=0) or -1 if missing
    return:
      split_code: int8 array length N, -1(drop) or
        0=train, 1=sel, 2=cal, 3=test
      meta: dict summary + family mapping info
    """
    rng = np.random.default_rng(seed)

    train_frac, sel_frac, cal_frac, test_frac = fracs
    s = train_frac + sel_frac + cal_frac + test_frac
    if not np.isclose(s, 1.0):
        raise ValueError(f"Fractions must sum to 1. Got sum={s}")

    N = fam_raw.size
    split_code = np.full(N, -1, dtype=np.int8)

    # drop missing family
    ok = fam_raw >= 0
    fam_ok = fam_raw[ok]
    if fam_ok.size == 0:
        raise ValueError("No samples have valid family_id (all missing).")

    # count families
    fam_vals, fam_counts = np.unique(fam_ok, return_counts=True)

    # drop small families
    keep_fam_mask = fam_counts >= min_family_count
    fam_keep_vals = fam_vals[keep_fam_mask]
    fam_keep_counts = fam_counts[keep_fam_mask]

    fam_keep_set = set(map(int, fam_keep_vals.tolist()))
    keep_idx = np.where(ok & np.isin(fam_raw, fam_keep_vals))[0]

    # reindex family ids to 0..F-1 (dense)
    fam_keep_vals_sorted = np.sort(fam_keep_vals)
    fam_to_new = {int(fid): i for i, fid in enumerate(fam_keep_vals_sorted)}
    fam_new = np.full(N, -1, dtype=np.int32)
    for i in keep_idx:
        fam_new[i] = fam_to_new[int(fam_raw[i])]

    F = len(fam_keep_vals_sorted)
    if F == 0:
        raise ValueError(f"No families survive min_family_count={min_family_count}.")

    # split within each family
    # codes: 0=train, 1=sel, 2=cal, 3=test
    n_train = n_sel = n_cal = n_test = 0

    for old_fid in fam_keep_vals_sorted:
        new_fid = fam_to_new[int(old_fid)]
        idx_f = np.where(fam_new == new_fid)[0]
        n = idx_f.size
        rng.shuffle(idx_f)

        # base allocations by fraction
        ns = int(np.floor(sel_frac * n))
        nc = int(np.floor(cal_frac * n))
        nt = int(np.floor(test_frac * n))
        # remaining goes to train
        nr = n - (ns + nc + nt)

        # enforce minimums (only if feasible)
        # we will move from train (nr) into sel/cal/test if needed.
        def bump_min(current, min_req):
            return max(current, min_req)

        # feasibility: at least 1 train left after mins
        # if impossible, we relax mins (keep proportional)
        min_total = min_sel_per_family + min_cal_per_family + min_test_per_family + 1
        if n >= min_total:
            ns2 = bump_min(ns, min_sel_per_family)
            nc2 = bump_min(nc, min_cal_per_family)
            nt2 = bump_min(nt, min_test_per_family)
            # adjust train remainder
            nr2 = n - (ns2 + nc2 + nt2)
            if nr2 < 1:
                # fallback: don't enforce mins
                ns2, nc2, nt2 = ns, nc, nt
                nr2 = n - (ns2 + nc2 + nt2)
            ns, nc, nt, nr = ns2, nc2, nt2, nr2

        # assign
        p = 0
        idx_test = idx_f[p:p+nt]; p += nt
        idx_cal  = idx_f[p:p+nc]; p += nc
        idx_sel  = idx_f[p:p+ns]; p += ns
        idx_tr   = idx_f[p:p+nr]; p += nr

        split_code[idx_tr] = 0
        split_code[idx_sel] = 1
        split_code[idx_cal] = 2
        split_code[idx_test] = 3

        n_train += idx_tr.size
        n_sel   += idx_sel.size
        n_cal   += idx_cal.size
        n_test  += idx_test.size

    meta = {
        "N_pool_total": int(N),
        "N_with_family": int(np.sum(ok)),
        "min_family_count": int(min_family_count),
        "N_kept": int(len(keep_idx)),
        "num_families_kept": int(F),
        "split_counts": {
            "train": int(n_train),
            "sel": int(n_sel),
            "cal": int(n_cal),
            "test": int(n_test),
        },
        "fractions": {
            "train": float(train_frac),
            "sel": float(sel_frac),
            "cal": float(cal_frac),
            "test": float(test_frac),
        },
        "min_per_family": {
            "sel": int(min_sel_per_family),
            "cal": int(min_cal_per_family),
            "test": int(min_test_per_family),
        },
        "family_old_ids_sorted": fam_keep_vals_sorted.astype(int).tolist(),
        # new id = index in family_old_ids_sorted
        "family_reindex_rule": "new_id = index in family_old_ids_sorted",
    }
    return split_code, fam_new, meta


def main():
    ap = argparse.ArgumentParser()

    # TFDS
    ap.add_argument("--data_dir", type=str, required=True, help="TFDS data_dir (e.g., data/tfds)")
    ap.add_argument("--pool_split", type=str, default="train+validation",
                    help='TFDS split expression used as pool (default: "train+validation")')

    # taxonomy mapping
    ap.add_argument("--cat_to_family_json", type=str, required=True, help="category_to_family.json")

    # output
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=224)

    # family filtering
    ap.add_argument("--min_family_count", type=int, default=250)

    # split fractions (Ding-style: all splits share same distribution)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--sel_frac", type=float, default=0.1)
    ap.add_argument("--cal_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.0)  # default 0 if you want no test; usually set 0.1
    # minimum per family (optional)
    ap.add_argument("--min_sel_per_family", type=int, default=0)
    ap.add_argument("--min_cal_per_family", type=int, default=0)
    ap.add_argument("--min_test_per_family", type=int, default=0)

    ap.add_argument("--save_indices", action="store_true", help="also store pool indices for each split")

    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    out_npz = os.path.expanduser(args.out_npz)
    cat_to_family_json = os.path.expanduser(args.cat_to_family_json)

    ensure_dir(os.path.dirname(out_npz))

    # load mapping species -> family
    sp_to_fam = load_cat_to_family(cat_to_family_json)
    print(f"[meta] loaded species->family map: {len(sp_to_fam)} / 5089 (species ids with family)")

    # build TFDS pool (deterministic order)
    ds_pool = tfds.load(
        "i_naturalist2017",
        data_dir=data_dir,
        split=args.pool_split,
        download=False,
        shuffle_files=False,
        as_supervised=False,
    )

    # -------------------------
    # pass1: scan labels -> family raw
    # -------------------------
    fam_raw = []
    sp_raw = []
    for ex in tqdm(tfds.as_numpy(ds_pool), desc="[pass1] scan pool labels -> family"):
        y = int(ex["label"])
        sp_raw.append(y)
        fam_raw.append(sp_to_fam.get(y, -1))
    sp_raw = np.asarray(sp_raw, dtype=np.int32)
    fam_raw = np.asarray(fam_raw, dtype=np.int32)

    # split codes
    fracs = (args.train_frac, args.sel_frac, args.cal_frac, args.test_frac)
    split_code, fam_new, split_meta = build_split_codes_by_family(
        fam_raw=fam_raw,
        min_family_count=args.min_family_count,
        seed=args.seed,
        fracs=fracs,
        min_sel_per_family=args.min_sel_per_family,
        min_cal_per_family=args.min_cal_per_family,
        min_test_per_family=args.min_test_per_family,
    )

    print("[pass1] done")
    print("[kept families]", split_meta["num_families_kept"])
    print("[split counts]", split_meta["split_counts"])

    # final split sizes
    n_train = int(np.sum(split_code == 0))
    n_sel   = int(np.sum(split_code == 1))
    n_cal   = int(np.sum(split_code == 2))
    n_test  = int(np.sum(split_code == 3))
    F = split_meta["num_families_kept"]

    # allocate arrays
    X_train = np.empty((n_train, 3, args.image_size, args.image_size), dtype=np.uint8)
    y_train = np.empty((n_train,), dtype=np.int64)
    X_sel   = np.empty((n_sel,   3, args.image_size, args.image_size), dtype=np.uint8)
    y_sel   = np.empty((n_sel,), dtype=np.int64)
    X_cal   = np.empty((n_cal,   3, args.image_size, args.image_size), dtype=np.uint8)
    y_cal   = np.empty((n_cal,), dtype=np.int64)
    X_test  = np.empty((n_test,  3, args.image_size, args.image_size), dtype=np.uint8)
    y_test  = np.empty((n_test,), dtype=np.int64)

    # -------------------------
    # pass2: collect images in ONE pass (fastest / consistent)
    # -------------------------
    ds_pool2 = tfds.load(
        "i_naturalist2017",
        data_dir=data_dir,
        split=args.pool_split,
        download=False,
        shuffle_files=False,
        as_supervised=False,
    )

    it_train = it_sel = it_cal = it_test = 0

    for i, ex in enumerate(tqdm(tfds.as_numpy(ds_pool2), total=fam_raw.size, desc="[pass2] collect images")):
        code = int(split_code[i])
        if code < 0:
            continue

        # decode + resize
        img = tf.convert_to_tensor(ex["image"])
        img = resize_uint8(img, args.image_size)
        x = img.numpy()  # HWC uint8
        x = np.transpose(x, (2, 0, 1))  # CHW

        yfam = int(fam_new[i])  # 0..F-1

        if code == 0:
            X_train[it_train] = x
            y_train[it_train] = yfam
            it_train += 1
        elif code == 1:
            X_sel[it_sel] = x
            y_sel[it_sel] = yfam
            it_sel += 1
        elif code == 2:
            X_cal[it_cal] = x
            y_cal[it_cal] = yfam
            it_cal += 1
        elif code == 3:
            X_test[it_test] = x
            y_test[it_test] = yfam
            it_test += 1

    # sanity
    if it_train != n_train or it_sel != n_sel or it_cal != n_cal or it_test != n_test:
        raise RuntimeError(
            f"Collected mismatch. train {it_train}/{n_train}, sel {it_sel}/{n_sel}, "
            f"cal {it_cal}/{n_cal}, test {it_test}/{n_test}"
        )

    # meta
    meta = {
        "tfds_data_dir": data_dir,
        "dataset": "i_naturalist2017",
        "pool_split": args.pool_split,
        "seed": int(args.seed),
        "image_size": int(args.image_size),
        "format": "uint8_CHW",
        "label_space": "family_reindexed_0..F-1",
        "F_num_families": int(F),
        "split_meta": split_meta,
        "note": (
            "Ding-style split: aggregate train+validation (pool_split), "
            "drop families with count < min_family_count, "
            "then split within each family so all splits share the same distribution."
        ),
    }

    save_kwargs = dict(
        X_train=X_train, y_train=y_train,
        X_sel=X_sel,     y_sel=y_sel,
        X_cal=X_cal,     y_cal=y_cal,
        X_test=X_test,   y_test=y_test,
        meta=np.array([meta], dtype=object),
    )

    if args.save_indices:
        idx_train = np.where(split_code == 0)[0].astype(np.int64)
        idx_sel   = np.where(split_code == 1)[0].astype(np.int64)
        idx_cal   = np.where(split_code == 2)[0].astype(np.int64)
        idx_test  = np.where(split_code == 3)[0].astype(np.int64)
        save_kwargs.update(
            idx_train=idx_train,
            idx_sel=idx_sel,
            idx_cal=idx_cal,
            idx_test=idx_test,
            pool_species_label=sp_raw.astype(np.int32),
            pool_family_raw=fam_raw.astype(np.int32),
            pool_family_new=fam_new.astype(np.int32),
            split_code=split_code.astype(np.int8),
        )

    np.savez_compressed(out_npz, **save_kwargs)
    print(f"[saved] {out_npz}")
    print("[shapes]",
          "X_train", X_train.shape, "y_train", y_train.shape,
          "X_sel",   X_sel.shape,   "y_sel",   y_sel.shape,
          "X_cal",   X_cal.shape,   "y_cal",   y_cal.shape,
          "X_test",  X_test.shape,  "y_test",  y_test.shape)


if __name__ == "__main__":
    main()
