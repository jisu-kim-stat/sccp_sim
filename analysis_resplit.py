import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. CSV 불러오기
df = pd.read_csv("out/results/resplit_seed1_k10.csv")

# 2. 평균 ± 표준편차 요약
metrics = [
    "marginal_cov",
    "avg_size",
    "avg_class_cov",
    "worst_class_cov",
    "maxgap",
]

methods = ["lccp", "cccp", "sccp"]

summary = {}

for m in methods:
    summary[m] = {}
    for met in metrics:
        col = f"{m}_{met}"
        summary[m][met] = f"{df[col].mean():.4f} ± {df[col].std():.4f}"

summary_df = pd.DataFrame(summary).T
print("\n=== Summary (mean ± std) ===")
print(summary_df)


# 3. worst_class_cov 박스플롯
df_plot = pd.melt(
    df,
    value_vars=[
        "lccp_worst_class_cov",
        "cccp_worst_class_cov",
        "sccp_worst_class_cov"
    ],
    var_name="method",
    value_name="worst_cov"
)

plt.figure(figsize=(6,4))
sns.boxplot(data=df_plot, x="method", y="worst_cov")
plt.title("Worst Class Coverage (20 reps)")
plt.show()