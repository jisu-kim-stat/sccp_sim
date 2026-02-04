import numpy as np
import sys

npz_path = sys.argv[1]
d = dict(np.load(npz_path, allow_pickle=True))

# 너 파일이 p_tst/y_tst이면 여기서 바꿔줘도 됨
p_cal = d.get("p_cal", None)
y_cal = d.get("y_cal", None)
p_test = d.get("p_test", None)
y_test = d.get("y_test", None)

if p_cal is None or y_cal is None:
    raise ValueError("Need p_cal & y_cal in npz")
if p_test is None or y_test is None:
    raise ValueError("Need p_test & y_test in npz")

p_cal = p_cal.astype(float)
y_cal = y_cal.astype(int)
p_test = p_test.astype(float)
y_test = y_test.astype(int)

def summarize_probs(p, y, name):
    N, K = p.shape
    true_p = p[np.arange(N), y]
    top1 = np.argmax(p, axis=1)
    top1_p = p[np.arange(N), top1]
    acc = float(np.mean(top1 == y))

    # 비확실성 지표들
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)  # natural log
    margin = top1_p - np.partition(p, -2, axis=1)[:, -2]           # top1 - top2
    score_true = 1.0 - true_p                                      # 네가 쓰는 nonconformity

    print(f"\n=== [{name}] ===")
    print("N,K:", N, K)
    print("top1 acc:", acc)
    print("true_p   mean/std:", float(true_p.mean()), float(true_p.std()))
    print("top1_p   mean/std:", float(top1_p.mean()), float(top1_p.std()))
    print("margin   mean/std:", float(margin.mean()), float(margin.std()))
    print("entropy  mean/std:", float(entropy.mean()), float(entropy.std()))
    print("score(1-true_p) mean/std:", float(score_true.mean()), float(score_true.std()))

    # 분포 분위수(분리 정도 감 잡기)
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    tq = np.quantile(true_p, qs)
    sq = np.quantile(score_true, qs)
    print("true_p quantiles   :", {q: float(v) for q, v in zip(qs, tq)})
    print("score quantiles    :", {q: float(v) for q, v in zip(qs, sq)})

    # “거의 구분 안됨”의 전형적 신호:
    # - top1_p가 낮고(예: 0.05~0.2), margin도 작고, entropy가 log(K)에 매우 가까움
    # - true_p 분포가 너무 한 점에 몰림(표준편차가 매우 작음)

summarize_probs(p_cal, y_cal, "CAL")
summarize_probs(p_test, y_test, "TEST")

# log(K) 비교: entropy가 이 값에 가까우면 거의 uniform 예측
K = p_cal.shape[1]
print("\nlog(K) =", float(np.log(K)), "(entropy가 이 근처면 거의 uniform)")
