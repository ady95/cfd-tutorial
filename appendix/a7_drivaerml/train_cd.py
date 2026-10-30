"""책 부록 A-7: DrivAerML의 형상 변수 16개로 항력 계수 cd를 예측한다.

데이터셋이 권장하는 분할을 그대로 쓴다.
- full / medium / scarce / super_scarce: train 400 / 133 / 67 / 11,
  val 34와 test 50은 공통 (train은 앞쪽이 뒤쪽에 포함된다)
- high_drag: cd가 가장 큰 20%를 test로 뺀 범위 밖 분할

비교 모델: 평균값, 선형 회귀, 2차 다항(릿지), XGBoost, 가우스 과정, MLP
오차는 항력 카운트(1 카운트 = cd 0.001)와 상대 오차로 본다.

사용법: uv run python appendix/a7_drivaerml/train_cd.py [데이터 폴더]
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def load(folder):
    geo = pd.read_csv(folder / "geo_parameters_all.csv", skipinitialspace=True)
    geo = geo.rename(columns={"Run": "run"})
    force = pd.read_csv(folder / "force_mom_all.csv")
    df = force.merge(geo, on="run")
    df.index = "run_" + df["run"].astype(str)
    feats = [c for c in geo.columns if c != "run"]
    return df, feats


class MLP:
    """5장과 같은 SiLU 64 x 3. 입력·출력을 표준화하고 val 오차가 가장 낮은 때를 쓴다."""

    def fit(self, x, y, xv, yv, epochs=3000, seed=0):
        torch.manual_seed(seed)
        self.xm, self.xs = x.mean(0), x.std(0)
        self.ym, self.ys = y.mean(), y.std()
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        X, Y = t((x - self.xm) / self.xs), t((y - self.ym) / self.ys)[:, None]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3,
                               weight_decay=1e-4)
        best, state = np.inf, None
        for ep in range(epochs):
            loss = torch.nn.functional.mse_loss(self.net(X), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ep % 10 == 0:
                v = np.abs(self.predict(xv) - yv).mean()
                if v < best:
                    best = v
                    state = {k: p.clone() for k, p in
                             self.net.state_dict().items()}
        self.net.load_state_dict(state)
        return self

    def predict(self, x):
        with torch.no_grad():
            z = self.net(torch.tensor((x - self.xm) / self.xs,
                                      dtype=torch.float32))
        return z.numpy().ravel() * self.ys + self.ym


def models():
    k = (ConstantKernel(1.0) * RBF(length_scale=np.ones(16),
                                   length_scale_bounds=(1e-2, 1e3))
         + WhiteKernel(1e-3))
    return {
        "평균값": None,
        "선형 회귀": make_pipeline(StandardScaler(), LinearRegression()),
        "2차 다항": make_pipeline(StandardScaler(), PolynomialFeatures(2),
                                RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.03,
                                max_depth=3, subsample=0.8, random_state=0),
        "가우스 과정": make_pipeline(StandardScaler(), GaussianProcessRegressor(
            k, normalize_y=True, n_restarts_optimizer=3, random_state=0)),
        "MLP": "mlp",
    }


def run(df, feats, split, manifest):
    tr, va, te = (df.loc[[r for r in manifest[f"{split}_{p}"]
                          if r in df.index]] for p in ("train", "val", "test"))
    x, y = tr[feats].to_numpy(), tr["cd"].to_numpy()
    xv, yv = va[feats].to_numpy(), va["cd"].to_numpy()
    xt, yt = te[feats].to_numpy(), te["cd"].to_numpy()
    rows = []
    for name, m in models().items():
        start = time.perf_counter()
        if m is None:
            p = np.full(len(yt), y.mean())
        elif m == "mlp":
            p = MLP().fit(x, y, xv, yv).predict(xt)
        else:
            p = m.fit(x, y).predict(xt)
        sec = time.perf_counter() - start
        e = np.abs(p - yt)
        rows.append((name, e.mean() * 1e3, e.max() * 1e3,
                     (e / yt).mean() * 100,
                     1 - ((p - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum(),
                     sec))
    return len(tr), len(te), rows


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1
                  else "~/cfd-runs/a7").expanduser()
    df, feats = load(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    print(f"공개 케이스 {len(df)}개, 형상 변수 {len(feats)}개, "
          f"cd {df.cd.min():.4f}~{df.cd.max():.4f} (평균 {df.cd.mean():.4f})")
    curve = {}
    for split in ["super_scarce", "scarce", "medium", "full", "high_drag"]:
        n_tr, n_te, rows = run(df, feats, split, manifest)
        print(f"\n[{split}] train {n_tr}, test {n_te}")
        print(f"{'모델':<10}{'MAE 카운트':>10}{'최대':>8}{'상대 %':>8}"
              f"{'R2':>8}{'학습 초':>8}")
        for name, mae, mx, relp, r2, sec in rows:
            print(f"{name:<10}{mae:10.2f}{mx:8.2f}{relp:8.2f}{r2:8.3f}"
                  f"{sec:8.1f}")
            if split != "high_drag":
                curve.setdefault(name, []).append((n_tr, mae))

    # 그림: train 개수에 따른 test MAE (같은 test 50개)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"평균값": "mean", "선형 회귀": "linear", "2차 다항": "quadratic",
             "XGBoost": "XGBoost", "가우스 과정": "Gaussian process",
             "MLP": "MLP"}
    colors = ["#888888", "#6c757d", "#e09f3e", "#d1495b", "#2a6f97",
              "#335c67"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for (name, pts), c in zip(curve.items(), colors):
        n, mae = zip(*pts)
        ax.plot(n, mae, "o-", color=c, lw=2, ms=6, label=names[name],
                ls="--" if name == "평균값" else "-")
    ax.set_xscale("log")
    ax.set_xticks([11, 67, 133, 400])
    ax.set_xticklabels(["11", "67", "133", "400"])
    ax.set_xlabel("training geometries (DrivAerML nested splits)")
    ax.set_ylabel("test MAE of cd (drag counts)")
    ax.set_ylim(0, 30)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(folder / "drivaerml-curve.png", dpi=130)
    print("\n저장:", folder / "drivaerml-curve.png")
