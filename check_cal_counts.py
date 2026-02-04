import numpy as np
import sys

npz_path = sys.argv[1]
d = dict(np.load(npz_path, allow_pickle=True))

y_cal = d["y_cal"].astype(int)
K = int(d["p_cal"].shape[1])

n_y = np.bincount(y_cal, minlength=K)

print("=== calibration label count summary ===")
print("K =", K)
print("N_cal =", y_cal.size)
print("min / median / mean / max n_y =",
      int(n_y.min()),
      int(np.median(n_y)),
      float(n_y.mean()),
      int(n_y.max()))

for t in [1, 2, 3, 5, 10, 20, 50]:
    print(f"#labels with n_y < {t}: {int(np.sum(n_y < t))}")

print("\n=== tail severity ===")
print("#labels with n_y == 0:", int(np.sum(n_y == 0)))
print("#labels with n_y == 1:", int(np.sum(n_y == 1)))
print("#labels with n_y <= 5:", int(np.sum(n_y <= 5)))

# 상위 몇 개만 보기
top_counts = np.sort(n_y)[-10:][::-1]
print("\nTop-10 largest class counts:", top_counts)
