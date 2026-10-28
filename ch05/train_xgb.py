"""책 05-3: XGBoost로 출력마다 회귀모델을 학습한다.

1) train에서 5겹 교차 검증과 조기 종료로 트리 개수를 정하고, train 전체로
   다시 학습해 val 상대 오차를 낸다.
2) 같은 학습을 CPU와 GPU로 해서 시간을 비교한다.
3) 출력별 특징 중요도(gain)를 출력한다.
4) train 크기를 줄여 가며 val 오차가 어떻게 변하는지(학습 곡선) 그린다.

사용법: uv run python ch05/train_xgb.py [출력 폴더]   (기본 ~/cfd-runs/ch05)
결과:   표 출력과 출력 폴더의 learning-curve.png
"""
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data import INPUTS, TARGETS, from_model, load, rel_error, to_model

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

PARAMS = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.05,
          "subsample": 0.8, "seed": 0}
MAX_ROUNDS, PATIENCE = 5000, 100


def cv_rounds(x, z, device="cpu"):
    """5겹 교차 검증에서 검증 오차가 가장 작아지는 트리 개수를 찾는다."""
    r = xgb.cv({**PARAMS, "device": device}, xgb.DMatrix(x, label=z),
               num_boost_round=MAX_ROUNDS, nfold=5, seed=0,
               early_stopping_rounds=PATIENCE)
    return len(r)


def fit(x, z, rounds, device="cpu"):
    return xgb.train({**PARAMS, "device": device}, xgb.DMatrix(x, label=z),
                     num_boost_round=rounds)


def fit_all(x, y, rounds=None):
    """출력마다 모델 하나씩. rounds가 없으면 교차 검증으로 정한다."""
    z = to_model(y)
    rounds = rounds or [cv_rounds(x, z[:, i]) for i in range(len(TARGETS))]
    return [fit(x, z[:, i], n) for i, n in enumerate(rounds)], rounds


def predict(models, x):
    d = xgb.DMatrix(x)
    return from_model(np.column_stack([m.predict(d) for m in models]))


x_tr, y_tr, _ = load("train")
x_va, y_va, _ = load("val")

# 1) 교차 검증으로 트리 개수를 정하고 val로 평가
models, rounds = fit_all(x_tr, y_tr)
e_tr, e_va = rel_error(y_tr, predict(models, x_tr)), rel_error(
    y_va, predict(models, x_va))
print("출력     트리 수  train 평균  val 평균  val 최대")
for t, n in zip(TARGETS, rounds):
    print(f"{t:<8} {n:6d}  {e_tr[t][0]:9.2f}%  {e_va[t][0]:7.2f}%"
          f"  {e_va[t][1]:7.1f}%")

# 2) CPU와 GPU의 학습 시간 (dp 하나, 교차 검증 + 최종 학습)
z = to_model(y_tr)[:, 0]
for device in ["cpu", "cuda"]:
    t = time.perf_counter()
    fit(x_tr, z, cv_rounds(x_tr, z, device), device)
    print(f"{device:<5} 학습 시간 {time.perf_counter() - t:5.1f}초")

# 3) 특징 중요도: 그 변수로 나눠서 줄어든 오차의 합(gain)의 비율
print("\n특징 중요도 [%]  " + "".join(f"{v:>9}" for v in INPUTS))
for t, m in zip(TARGETS, models):
    g = m.get_score(importance_type="total_gain")
    total = sum(g.values())
    print(f"{t:<16} " + "".join(f"{g.get(f'f{i}', 0) / total * 100:9.1f}"
                                for i in range(len(INPUTS))))

# 4) 학습 곡선: train에서 n개를 무작위로 5번 뽑아 학습하고 val 오차 평균
sizes, reps = [50, 100, 200, 300, 418], 5
curve = {"XGBoost": [], "cubic polynomial": []}
for n in sizes:
    err = {k: [] for k in curve}
    for s in range(reps):
        idx = np.random.default_rng(s).permutation(len(x_tr))[:n]
        xs, ys = x_tr[idx], y_tr[idx]
        m, _ = fit_all(xs, ys, rounds)
        err["XGBoost"].append(rel_error(y_va, predict(m, x_va)))
        p = make_pipeline(StandardScaler(), PolynomialFeatures(3),
                          LinearRegression()).fit(xs, to_model(ys))
        err["cubic polynomial"].append(
            rel_error(y_va, from_model(p.predict(x_va))))
    for k in curve:     # 세 출력과 반복의 평균 상대 오차
        curve[k].append(np.mean([[e[t][0] for t in TARGETS]
                                 for e in err[k]]))
print("\n학습 데이터 수   " + "".join(f"{n:>7}" for n in sizes))
for k, v in curve.items():
    print(f"{k:<16} " + "".join(f"{e:6.2f}%" for e in v))

out = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/ch05")
out = out.expanduser()
out.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4.2))
for (k, v), c, mk in zip(curve.items(), ["#2a78d6", "#eb6834"], "os"):
    ax.loglog(sizes, v, mk + "-", color=c, lw=2, ms=8, label=k)
ax.axhline(1.0, ls="--", color="0.5", lw=1.5, label="grid error (about 1%)")
ax.set_xticks(sizes, [str(n) for n in sizes])
ax.set_yticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
ax.minorticks_off()
ax.set(xlabel="number of training cases",
       ylabel="mean relative error on val [%]",
       title="learning curve (average of dp, dT_max, dT_avg)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(out / "learning-curve.png", dpi=150)
