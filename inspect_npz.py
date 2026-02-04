import numpy as np
import sys

npz_path = sys.argv[1]
d = dict(np.load(npz_path, allow_pickle=True))

print("=== keys ===")
for k in sorted(d.keys()):
    v = d[k]
    if isinstance(v, np.ndarray):
        print(f"{k:>12s} | shape={v.shape} dtype={v.dtype}")
    else:
        print(f"{k:>12s} | type={type(v)}")

# 핵심 키 후보들
candidates = [
    ("p_cal", "y_cal"),
    ("p_test", "y_test"),
    ("p_tst", "y_tst"),
    ("p_sel", "y_sel"),
]
print("\n=== key existence check ===")
for pk, yk in candidates:
    print(f"{pk},{yk}:", (pk in d), (yk in d))

# p_cal/p_test 있으면 간단 sanity check
def check_probs(name):
    p = d[name]
    p = p.astype(float)
    print(f"\n[{name}] N={p.shape[0]} K={p.shape[1]}")
    print("  min/max:", np.min(p), np.max(p))
    rs = p.sum(axis=1)
    print("  row sum | mean(abs(sum-1)) =", float(np.mean(np.abs(rs-1))))
    print("  row sum | min/max =", float(rs.min()), float(rs.max()))
    print("  nan?", np.isnan(p).any(), "inf?", np.isinf(p).any())

if "p_cal" in d: check_probs("p_cal")
if "p_test" in d: check_probs("p_test")
if "p_tst" in d: check_probs("p_tst")
