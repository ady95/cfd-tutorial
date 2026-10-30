"""책 부록 A-6: 평판 데이터셋으로 응력 집중 계수 Kt와 유연도 C의 대리모델을 만든다.

비교하는 모델 (5장과 같은 순서)
- 공학 공식 (Heywood): 폭이 유한하고 높이가 무한한 판의 Kt. C는 없음
- 3차 다항 회귀
- MLP (SiLU 64 x 3, 5장과 같은 구조), train 개수를 25, 50, 100으로 바꿔 가며

사용법: uv run python appendix/a6_structure/train_plate.py [plates.csv]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

IN, OUT = ["r", "h", "nu"], ["Kt", "C"]
SPLITS = ["val", "test", "ood_r"]


def heywood(r):
    """Heywood 식: Kt(순단면) = 2 + (1 - d/W)^3 를 전체 단면 기준으로 바꾼 값."""
    return (2 + (1 - r) ** 3) / (1 - r)


def train_mlp(x, y, epochs=5000, seed=0):
    torch.manual_seed(seed)
    xm, xs = x.mean(0), x.std(0)
    ym, ys = y.mean(0), y.std(0)
    X = torch.tensor((x - xm) / xs, dtype=torch.float32)
    Y = torch.tensor((y - ym) / ys, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(3, 64), torch.nn.SiLU(),
        torch.nn.Linear(64, 64), torch.nn.SiLU(),
        torch.nn.Linear(64, 64), torch.nn.SiLU(),
        torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-5)
    for _ in range(epochs):
        loss = torch.nn.functional.mse_loss(net(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    def predict(xn):
        with torch.no_grad():
            z = net(torch.tensor((xn - xm) / xs, dtype=torch.float32))
        return z.numpy() * ys + ym
    return predict


def rel(p, t):
    return np.abs(p / t - 1) * 100


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                Path(__file__).resolve().parent / "plates.csv")
    df = pd.read_csv(path)
    tr = df[df.split == "train"]
    x, y = tr[IN].to_numpy(), tr[OUT].to_numpy()

    models = {"Heywood 공식": None}
    poly = make_pipeline(StandardScaler(), PolynomialFeatures(3),
                         LinearRegression()).fit(x, y)
    models["3차 다항"] = poly.predict
    for n in (25, 50, 100):
        start = time.perf_counter()
        models[f"MLP train {n}"] = train_mlp(x[:n], y[:n])
        print(f"MLP train {n}개 학습 {time.perf_counter() - start:.1f}초")

    print("\n상대 오차 [%] (평균/최대)")
    print(f"{'모델':<14}{'split':>6}{'Kt':>16}{'C':>16}")
    for name, f in models.items():
        for s in SPLITS:
            d = df[df.split == s]
            t = d[OUT].to_numpy()
            if f is None:
                e = rel(heywood(d["r"].to_numpy()), t[:, 0])
                cols = [f"{e.mean():6.2f}/{e.max():6.2f}", "-"]
            else:
                e = rel(f(d[IN].to_numpy()), t)
                cols = [f"{e[:, k].mean():6.3f}/{e[:, k].max():6.3f}"
                        for k in range(2)]
            print(f"{name:<14}{s:>6}" + "".join(f"{c:>16}" for c in cols))

    # 판 높이에 따른 공식 오차 (test + val)
    d = df[df.split.isin(["val", "test"])]
    e = rel(heywood(d["r"].to_numpy()), d["Kt"].to_numpy())
    for lo, hi in [(1.0, 1.33), (1.33, 1.67), (1.67, 2.0)]:
        k = (d["h"] >= lo) & (d["h"] < hi + 1e-9)
        print(f"h {lo:.2f}~{hi:.2f}: Heywood 평균 오차 {e[k].mean():.2f}% "
              f"({k.sum()}개)")

    # 그림: 해석 Kt (판 높이별 색)와 Heywood 공식
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path("~/cfd-runs/a6").expanduser()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    inr = df[df.split != "ood_r"]
    sc = ax.scatter(inr["r"], inr["Kt"], c=inr["h"], cmap="viridis_r",
                    s=22, edgecolors="white", linewidths=0.5)
    r = np.linspace(0.1, 0.45, 100)
    ax.plot(r, heywood(r), color="#333333", lw=2,
            label="Heywood formula (infinite height)")
    ood = df[df.split == "ood_r"]
    ax.scatter(ood["r"], ood["Kt"], marker="x", color="#d1495b", s=30,
               label="ood_r (larger holes)")
    fig.colorbar(sc, ax=ax, label="plate height ratio h = Hp/W")
    ax.set_xlabel("hole radius ratio r = R/W")
    ax.set_ylabel("stress concentration factor Kt")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "plate-kt.png", dpi=130)
    print("저장:", out / "plate-kt.png")
