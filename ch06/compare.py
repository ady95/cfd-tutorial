"""책 06-5: 모델을 한 표로 비교하고, 최종 MLP의 오차가 큰 곳을 찾는다.

1) 범위 안 데이터(train+val+test) 전체를 5겹으로 나눠 MLP를 학습하고,
   모든 케이스의 out-of-fold 오차를 구한다. test는 06-1에서 최종 평가에
   이미 썼으므로 여기서는 오차 분석에 함께 쓴다.
2) 평균값·선형·3차·6차 다항·XGBoost·MLP를 train으로 학습해 학습 시간,
   test와 OOD의 상대 오차, 한 건 추론 시간을 한 표로 정리한다.
3) 1)의 오차가 큰 케이스가 설계 공간 어디에 모여 있는지 보고, 입력마다
   오차의 구간 평균을 그림으로 남긴다.

사용법: uv run python ch06/compare.py [출력 폴더]   (기본 ~/cfd-runs/ch06)
        CPU 6코어 기준 약 1분
결과:   표 출력과 출력 폴더의 error-by-input.png
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from common import (INPUTS, TARGETS, fit_mean, fit_mlp, fit_poly,
                    fit_xgb, load, metrics)

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

FITS = {"평균값": fit_mean, "선형 회귀": lambda x, y: fit_poly(x, y, 1),
        "3차 다항": lambda x, y: fit_poly(x, y, 3),
        "6차 다항": lambda x, y: fit_poly(x, y, 6),
        "XGBoost": fit_xgb, "MLP": fit_mlp}


def width(s):
    return sum(2 if ord(c) > 0x1100 else 1 for c in s)   # 한글은 2칸


def mean_rel(y, p):
    """세 출력 평균 상대 오차의 평균과, 세 출력 통틀어 최댓값 [%]."""
    m = metrics(y, p)
    return (np.mean([v["rel_mean"] for v in m.values()]),
            max(v["rel_max"] for v in m.values()))


# 1) 범위 안 전체의 out-of-fold 오차. PyTorch는 부모 프로세스에서 한 번 학습한
#    뒤에는 fork로 만든 자식 프로세스에서 학습하지 못하므로 이것부터 계산한다
def run_fold(fold):
    torch.set_num_threads(1)            # 5개를 동시에 돌리므로 코어 하나씩
    train, test = fold
    return fit_mlp(x_all[train], y_all[train])(x_all[test])


x_all, y_all, df = load("train", "val", "test")
folds = list(KFold(5, shuffle=True, random_state=0).split(x_all))
with ProcessPoolExecutor(5) as pool:
    preds = list(pool.map(run_fold, folds))
p_all = np.empty_like(y_all)
for (_, test), p in zip(folds, preds):
    p_all[test] = p
err = np.abs(p_all / y_all - 1) * 100
df["err"] = err.max(axis=1)             # 세 출력 중 가장 나쁜 오차

# 2) 종합 비교표
x_tr, y_tr, _ = load("train")
sets = {s: load(s)[:2] for s in ("test", "ood_U", "ood_H")}
one = x_tr[:1]
print("모델        학습 시간   test 평균  test 최대  ood_U 평균  ood_H 평균"
      "  추론 1건")
for name, fit in FITS.items():
    start = time.perf_counter()
    f = fit(x_tr, y_tr)
    sec = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(200):
        f(one)
    t_one = (time.perf_counter() - start) / 200
    (te, te_max), (u, _), (h, _) = (mean_rel(y, f(x)) for x, y in sets.values())
    print(name + " " * (12 - width(name)) + f"{sec:7.2f}초 {te:9.2f}% "
          f"{te_max:8.1f}% {u:10.2f}% {h:10.2f}% {t_one * 1e3:7.2f}ms")

# 3) 오차가 큰 케이스는 어디에 있나
print(f"\n범위 안 {len(df)}개 out-of-fold 평균 상대 오차: " + ", ".join(
    f"{t} {e:.3f}%" for t, e in zip(TARGETS, err.mean(axis=0))))
top = df[df["err"] >= df["err"].quantile(0.9)]
print(f"\n오차 상위 10% ({len(top)}개, 세 출력 중 최대 오차 "
      f"{top['err'].min():.2f}% 이상)의 입력 평균과 전체 평균")
print("            " + "".join(f"{c:>9}" for c in INPUTS) + "  반복 수 중앙값")
for name, d in [("상위 10%", top), ("전체", df)]:
    print(name + " " * (12 - width(name))
          + "".join(f"{d[c].mean():9.3f}" for c in INPUTS)
          + f"  {d['iterations'].median():9.0f}")

corner = (df["e_ratio"] > 0.35) & (df["U"] > 0.12)
print(f"\ne/H > 0.35이고 U > 0.12인 모서리: 전체의 {corner.mean():.0%},"
      f" 오차 상위 10%의 {corner[top.index].mean():.0%}")
for name, k in [("모서리", corner), ("나머지", ~corner)]:
    print(f"  {name} {k.sum():3d}개: 세 출력 중 최대 오차의 평균 "
          f"{df.loc[k, 'err'].mean():.3f}%, "
          f"최댓값 {df.loc[k, 'err'].max():.2f}%")
slow = df["iterations"] > 1000
print(f"수렴에 1000회 넘게 걸린 케이스 {slow.sum()}개: 평균 "
      f"{df.loc[slow, 'err'].mean():.3f}% / 나머지 "
      f"{df.loc[~slow, 'err'].mean():.3f}%")

# 그림: 입력마다 오차와 구간 평균
out = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/ch06")
out = out.expanduser()
out.mkdir(parents=True, exist_ok=True)
names = ["channel height H [mm]", "rib height ratio e/H", "number of ribs N",
         "inlet velocity U [m/s]"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for i, (ax, c, name) in enumerate(zip(axes, INPUTS, names)):
    ax.scatter(df[c], df["err"], s=8, color="0.65", label="each case")
    k = pd.qcut(df[c], 6, labels=False, duplicates="drop")  # 개수가 같은 6구간
    g = df.groupby(k)
    ax.plot(g[c].mean(), g["err"].mean(), "o-", color="#2a78d6", lw=2,
            label="bin mean")
    ax.set(xlabel=name, yscale="log", title=f"({'abcd'[i]}) error vs. {c}")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("max relative error of 3 outputs [%]")
axes[0].legend(loc="lower left")
fig.tight_layout()
fig.savefig(out / "error-by-input.png", dpi=150)
