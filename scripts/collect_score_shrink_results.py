import json
import glob
import numpy as np
import pandas as pd

FILES = sorted(glob.glob("out/results/score_shrink_lccp_seed*.json"))
if not FILES:
    raise SystemExit("No files matched: out/results/score_shrink_lccp_seed*.json")

def safe_get(d, path, default=np.nan):
    """path: list of keys"""
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def pick_seed_from_npz(npz_path: str):
    # e.g., data/npz/inat2017_probs_family_resnet50_seed1.npz
    import re
    m = re.search(r"seed(\d+)", npz_path)
    return int(m.group(1)) if m else None

rows = []

for f in FILES:
    with open(f, "r") as fp:
        d = json.load(fp)

    npz_path = d.get("npz", "")
    seed = pick_seed_from_npz(npz_path)  # 안정적으로 seed 추출

    best_tau = safe_get(d, ["tuning", "best_tau"], default=np.nan)

    for tag, root in [
        ("baseline_LCCP", ["baseline_LCCP"]),
        ("proposed_D2", ["proposed_ScoreShrink_LCCP_D2"]),
        ("tuned_bestTau(sel)", ["tuning", "sel_metrics_best_tau"]),
    ]:
        rows.append({
            "seed": seed,
            "variant": tag,
            "best_tau": best_tau if tag == "tuned_bestTau(sel)" else np.nan,
            "marginal_cov": safe_get(d, root + ["marginal_cov"]),
            "avg_size": safe_get(d, root + ["avg_size"]),
            "worst_class_cov": safe_get(d, root + ["worst_class_cov"]),
            "avg_class_cov": safe_get(d, root + ["avg_class_cov"]),
            "std_class_cov": safe_get(d, root + ["std_class_cov"]),
            "cov_head": safe_get(d, root + ["cov_head"]),
            "cov_tail": safe_get(d, root + ["cov_tail"]),
            "size_head": safe_get(d, root + ["size_head"]),
            "size_tail": safe_get(d, root + ["size_tail"]),
            "n_head": safe_get(d, root + ["n_head"]),
            "n_tail": safe_get(d, root + ["n_tail"]),
            "covgap": safe_get(d, root + ["covgap"]),
            "maxgap": safe_get(d, root + ["maxgap"]),
            "file": f,
        })

df = pd.DataFrame(rows)
df = df.sort_values(["variant", "seed"]).reset_index(drop=True)

# 저장 (long format)
out_csv = "out/results/score_shrink_all_seeds_long.csv"
df.to_csv(out_csv, index=False)

print("\n=== Per-seed (long) ===")
print(df[["seed","variant","best_tau","marginal_cov","avg_size","cov_tail","size_tail","worst_class_cov"]])

# 요약 (variant별 mean±std)
num_cols = [
    "marginal_cov","avg_size","worst_class_cov","avg_class_cov","std_class_cov",
    "cov_head","cov_tail","size_head","size_tail","covgap","maxgap"
]
summary_mean = df.groupby("variant")[num_cols].mean(numeric_only=True)
summary_std  = df.groupby("variant")[num_cols].std(numeric_only=True)

# 저장 (wide summary)
summary_mean.to_csv("out/results/score_shrink_summary_mean.csv")
summary_std.to_csv("out/results/score_shrink_summary_std.csv")

print("\n=== Summary (mean ± std) by variant ===")
for v in summary_mean.index:
    mc_m, mc_s = summary_mean.loc[v,"marginal_cov"], summary_std.loc[v,"marginal_cov"]
    sz_m, sz_s = summary_mean.loc[v,"avg_size"], summary_std.loc[v,"avg_size"]
    tl_m, tl_s = summary_mean.loc[v,"cov_tail"], summary_std.loc[v,"cov_tail"]
    wz_m, wz_s = summary_mean.loc[v,"worst_class_cov"], summary_std.loc[v,"worst_class_cov"]
    bt = df.loc[df["variant"].eq(v), "best_tau"].dropna()
    bt_str = f" | best_tau mean={bt.mean():.3g}" if len(bt) else ""
    print(f"- {v}{bt_str}")
    print(f"  marginal_cov: {mc_m:.4f} ± {mc_s:.4f}")
    print(f"  avg_size    : {sz_m:.3f} ± {sz_s:.3f}")
    print(f"  cov_tail    : {tl_m:.4f} ± {tl_s:.4f}")
    print(f"  worst_class : {wz_m:.4f} ± {wz_s:.4f}")

print(f"\n[Saved] {out_csv}")
print("[Saved] out/results/score_shrink_summary_mean.csv")
print("[Saved] out/results/score_shrink_summary_std.csv")
