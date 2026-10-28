"""책 05-2: 단순 회귀모델로 기준선을 만든다.

평균값, 선형 회귀, 거듭제곱 법칙(로그-로그 선형 회귀), 2~6차 다항 회귀를
train으로 학습해 train과 val의 상대 오차를 비교하고, 잔차 그림을 그린다.

사용법: uv run python ch05/baseline.py [출력 폴더]   (기본 ~/cfd-runs/ch05)
결과:   표 출력과 출력 폴더의 baseline-residual.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data import from_model, load, rel_error, to_model

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402


class Mean:
    """train 평균을 언제나 그대로 내놓는 가장 단순한 예측."""

    def fit(self, x, z):
        self.m = z.mean(axis=0)
        return self

    def predict(self, x):
        return np.tile(self.m, (len(x), 1))


def poly(degree):
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree),
                         LinearRegression())


x_tr, y_tr, _ = load("train")
x_va, y_va, _ = load("val")

# (이름, 모델, 입력 변환, 출력 변환, 출력 역변환)
same = (lambda x: x)
models = [
    ("평균값", Mean(), same, to_model, from_model),
    ("선형 회귀", LinearRegression(), same, to_model, from_model),
    ("거듭제곱 법칙", LinearRegression(), np.log, np.log, np.exp),
] + [(f"{d}차 다항 회귀", poly(d), same, to_model, from_model)
     for d in range(2, 7)]


def width(s):
    return sum(2 if ord(c) > 0x1100 else 1 for c in s)   # 한글은 2칸


print("모델              평균 상대 오차 [%] train / val          val 최대 [%]")
print(" " * 18 + "dp          dT_max      dT_avg" + " " * 11 + "dp dT_max dT_avg")
for name, m, fx, fy, inv in models:
    m.fit(fx(x_tr), fy(y_tr))
    tr = rel_error(y_tr, inv(m.predict(fx(x_tr))))
    va = rel_error(y_va, inv(m.predict(fx(x_va))))
    line = name + " " * (18 - width(name))
    line += "".join(f"{tr[t][0]:5.2f}/{va[t][0]:5.2f} " for t in tr)
    print(line + " " + "".join(f"{va[t][1]:6.1f} " for t in va).rstrip())

# 잔차 그림: 최고 온도 상승의 상대 오차를 유속에 대해
out = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/ch05")
out = out.expanduser()
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, name in zip(axes, ["선형 회귀", "거듭제곱 법칙", "3차 다항 회귀"]):
    m, fx, fy, inv = next(r[1:] for r in models if r[0] == name)
    err = (inv(m.predict(fx(x_tr)))[:, 1] / y_tr[:, 1] - 1) * 100
    sc = ax.scatter(x_tr[:, 3], err, c=x_tr[:, 1], s=12, cmap="viridis")
    ax.axhline(0, color="0.3", lw=1)
    ax.set(xlabel="inlet velocity U [m/s]",
           title={"선형 회귀": "(a) linear", "거듭제곱 법칙": "(b) power law",
                  "3차 다항 회귀": "(c) cubic polynomial"}[name])
    ax.grid(alpha=0.3)
axes[0].set_ylabel("dT_max error on train [%]")
fig.colorbar(sc, ax=axes, label="rib height ratio e/H")
fig.savefig(out / "baseline-residual.png", dpi=150, bbox_inches="tight")
