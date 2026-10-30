"""책 11-1: 예측 불확실성을 세 가지 방법으로 추정하고 실제 오차와 비교한다.

- 앙상블: 초깃값만 다른 MLP 5개 예측의 표준편차
- MC 드롭아웃: 드롭아웃을 켠 채 같은 MLP로 100번 예측한 표준편차
- 가우시안 프로세스(GP): 출력마다 Matern 커널 GP의 예측 표준편차
train 50개와 418개(전체)로 각각 학습하고, 새 시험 데이터(범위 안 200개)와
범위 밖 데이터(ood_U, ood_H 100개)에서 불확실성이 오차를 얼마나 따라가는지 본다.

사용법: uv run python ch11/uncertainty.py   (CPU 약 4분)
결과 그림: ~/cfd-runs/ch11/uncertainty.png
"""
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from torch import nn

from common import (INPUTS, LOG, TARGETS, Scaler, case_error, fit_ensemble,
                    from_model, load, pad, summary, to_model)
from new_test import load_new_test

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("~/cfd-runs/ch11").expanduser()
SIZES = [50, 418]          # train의 수렴한 케이스는 418개
# GP 하이퍼파라미터 최적화가 경계에 닿았다는 경고는 결과에 영향이 없어 끈다
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit_dropout(x, y, p=0.05, samples=100, seed=0, epochs=20000):
    """드롭아웃을 넣은 05-4 MLP. 예측할 때도 드롭아웃을 켜 둔다."""
    sx, sz = Scaler().fit(x), Scaler().fit(to_model(y))
    xt = torch.tensor(sx.transform(x), dtype=torch.float32)
    zt = torch.tensor(sz.transform(to_model(y)), dtype=torch.float32)
    torch.manual_seed(seed)
    layers, n = [], len(INPUTS)
    for _ in range(3):
        layers += [nn.Linear(n, 64), nn.SiLU(), nn.Dropout(p)]
        n = 64
    model = nn.Sequential(*layers, nn.Linear(n, len(TARGETS)))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad()
        ((model(xt) - zt) ** 2).mean().backward()
        opt.step()
        sched.step()

    def predict(x_new):
        xs = torch.tensor(sx.transform(x_new), dtype=torch.float32)
        with torch.no_grad():       # model.train() 상태 그대로: 드롭아웃이 켜짐
            ys = np.stack([from_model(sz.inverse(model(xs).numpy()
                                                  .astype(float)))
                           for _ in range(samples)])
        mean = ys.mean(axis=0)
        return mean, ys.std(axis=0) / mean * 100
    return predict


def fit_gp(x, y):
    """출력마다 GP 하나. 학습용 값(로그 dp 포함)에서 맞추고 % 불확실성으로."""
    sx, sz = Scaler().fit(x), Scaler().fit(to_model(y))
    xs, zs = sx.transform(x), sz.transform(to_model(y))
    kernel = (ConstantKernel() * Matern(length_scale=np.ones(4), nu=2.5)
              + WhiteKernel(1e-4, (1e-8, 1e-1)))
    gps = [GaussianProcessRegressor(kernel, n_restarts_optimizer=3,
                                    random_state=0).fit(xs, zs[:, i])
           for i in range(len(TARGETS))]

    def predict(x_new):
        out = [g.predict(sx.transform(x_new), return_std=True) for g in gps]
        z = np.column_stack([m for m, _ in out])
        s = np.column_stack([s for _, s in out]) * sz.std
        mean = from_model(sz.inverse(z))
        # 로그 dp의 표준편차는 그대로 상대값, 온도는 예측값으로 나눈다
        rel = np.where(LOG, s, s / mean) * 100
        return mean, rel
    return predict


def evaluate(predict, x_val, y_val, x_in, y_in, x_out, y_out):
    """보정 배율(val에서), 범위 안·밖의 오차·불확실성·순위 상관·포함률."""
    mv, uv = predict(x_val)
    ev = np.abs(mv / y_val - 1) * 100
    k = np.quantile(ev / uv, 0.95) / 2          # 2kσ 안에 95%가 들도록
    res = {"k": k}
    for name, x, y in (("in", x_in, y_in), ("out", x_out, y_out)):
        m, u = predict(x)
        e = np.abs(m / y - 1) * 100
        res[name] = {"err": case_error(y, m).mean(), "u": summary(u).mean(),
                     "rho": spearmanr(summary(u), case_error(y, m))[0],
                     "cover": (e <= 2 * u).mean() * 100,
                     "cover_k": (e <= 2 * k * u).mean() * 100,
                     "u_case": summary(u), "e_case": case_error(y, m)}
    return res


def main():
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, _ = load("train")
    x_val, y_val, _ = load("val")
    x_in, y_in = load_new_test()
    x_out, y_out, _ = load("ood_U", "ood_H")
    print(f"train {len(x)}개, val {len(x_val)}개, 새 시험 {len(x_in)}개, "
          f"범위 밖 {len(x_out)}개")
    order = np.random.default_rng(0).permutation(len(x))

    methods = {"앙상블 5개": lambda a, b: fit_ensemble(a, b).predict,
               "MC 드롭아웃": fit_dropout, "GP": fit_gp}
    keep, single, sec = {}, {}, {}
    for n in SIZES:
        idx = order[:n]
        # 비교 기준: 불확실성 없이 MLP 하나 (05-4와 같은 학습)
        t = time.time()
        one = fit_ensemble(x[idx], y[idx], members=1)
        sec[n, "MLP 1개"] = time.time() - t
        single[n] = [case_error(ys, one.predict(xs)[0]).mean()
                     for xs, ys in ((x_in, y_in), (x_out, y_out))]
        for name, fit in methods.items():
            t = time.time()
            predict = fit(x[idx], y[idx])
            sec[n, name] = time.time() - t
            keep[n, name] = evaluate(predict, x_val, y_val, x_in, y_in,
                                     x_out, y_out)

    print("\n[범위 안: 새 시험 200개]")
    print("데이터  방법          학습 시간    오차   불확실성  순위 상관  2σ 포함")
    for n in SIZES:
        print(f"{n:4d}개  {pad('MLP 1개', 12)}{sec[n, 'MLP 1개']:7.1f}초"
              f" {single[n][0]:7.3f}%")
        for name in methods:
            a = keep[n, name]["in"]
            print(f"{n:4d}개  {pad(name, 12)}{sec[n, name]:7.1f}초"
                  f" {a['err']:7.3f}% {a['u']:8.3f}% {a['rho']:9.2f}"
                  f" {a['cover']:7.0f}%")

    print("\n[범위 밖: ood_U, ood_H] 보정 배율은 val에서 2kσ 안에 95%가 들게 정한 k")
    print("데이터  방법             오차   불확실성  순위 상관  보정 배율"
          "  보정 후(안/밖)")
    for n in SIZES:
        print(f"{n:4d}개  {pad('MLP 1개', 12)}{single[n][1]:8.3f}%")
        for name in methods:
            r = keep[n, name]
            b = r["out"]
            print(f"{n:4d}개  {pad(name, 12)}{b['err']:8.3f}% {b['u']:8.3f}%"
                  f" {b['rho']:9.2f} {r['k']:9.1f}"
                  f" {r['in']['cover_k']:8.0f}% / {b['cover_k']:.0f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True,
                             sharey=True)
    for ax, n in zip(axes, SIZES):
        r = keep[n, "앙상블 5개"]
        for key, label, c in (("in", "in range (new test, 200)", "#2a6fdb"),
                              ("out", "out of range (ood_U, ood_H)", "#e8710a")):
            ax.scatter(r[key]["u_case"], r[key]["e_case"], s=14, c=c,
                       alpha=0.7, label=label, edgecolors="none")
        lim = np.array([1e-3, 30])
        ax.plot(lim, lim, "k--", lw=1, label="error = uncertainty")
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
               title=f"ensemble of 5, trained on {n}",
               xlabel="uncertainty (std. across members) [%]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("actual relative error [%]")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "uncertainty.png", dpi=150)
    print(f"\n그림: {OUT / 'uncertainty.png'}")


if __name__ == "__main__":
    main()
