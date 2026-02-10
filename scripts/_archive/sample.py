import os
import numpy as np
import matplotlib.pyplot as plt

npz_path = "/home/jisukim/sccp_sim/cifar100/out/cifar100LT_probs/cifar100_resnet18_e100_bs128_LT_exp_IF100_tailfrac0.2.npz"
z = np.load(npz_path, allow_pickle=True)

P = z["p_tst"]   # (n, K)
y = z["y_tst"]   # (n,)
n, K = P.shape

print("[loaded]", npz_path)
print("[shapes]", "P:", P.shape, "y:", y.shape)

# true prob
p_true = P[np.arange(n), y]

# false prob (sample to avoid massive size)
mask = np.ones_like(P, dtype=bool)
mask[np.arange(n), y] = False
p_false_all = P[mask]  # length n*(K-1)

rng = np.random.default_rng(0)
m = min(200_000, p_false_all.size)  # sample size for plotting
p_false = rng.choice(p_false_all, size=m, replace=False)

# summaries
def q(a, qs=(0.01, 0.05, 0.5, 0.95, 0.99)):
    return {f"q{int(qq*100):02d}": float(np.quantile(a, qq)) for qq in qs}

print("\n[true prob summary]")
print(" min/mean/max:", float(p_true.min()), float(p_true.mean()), float(p_true.max()))
print(" quantiles:", q(p_true))

print("\n[false prob summary] (sampled)")
print(" min/mean/max:", float(p_false.min()), float(p_false.mean()), float(p_false.max()))
print(" quantiles:", q(p_false))

# plot
out_png = os.path.splitext(npz_path)[0] + "_true_vs_false.png"

plt.figure(figsize=(7, 4.2))
plt.hist(p_false, bins=80, density=True, alpha=0.65, label="False class prob (sampled)")
plt.hist(p_true,  bins=80, density=True, alpha=0.75, label="True class prob")
plt.xlabel("Predicted probability")
plt.ylabel("Density")
plt.title("CIFAR100-LT (ResNet18 e100): True vs False class probabilities")
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()

print("\n[saved figure]", out_png)
