#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    z = np.load(args.in_npz, allow_pickle=True)

    out = dict(
        p_sel=z["p_sel"].astype(np.float32),
        y_sel=z["y_sel"].astype(np.int64),
        p_cal=z["p_cal"].astype(np.float32),
        y_cal=z["y_cal"].astype(np.int64),

        # 핵심: tst -> test로 rename
        p_test=z["p_tst"].astype(np.float32),
        y_test=z["y_tst"].astype(np.int64),
    )

    # 유용한 메타/테일정보는 그대로 복사
    for k in ["tail_set","counts_pool","counts_tr","counts_sel","counts_cal","counts_test","meta_json",
              "idx_tr","idx_sel","idx_cal","idx_train_pool","idx_test"]:
        if k in z.files:
            out[k] = z[k]

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, **out)

    print("[saved]", args.out_npz)
    print("[keys]", list(out.keys()))
    print("[shapes] p_sel", out["p_sel"].shape, "p_cal", out["p_cal"].shape, "p_test", out["p_test"].shape)

if __name__ == "__main__":
    main()
