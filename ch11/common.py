"""책 11장 공통: 예측과 함께 불확실성을 내놓는 대리모델과 샘플 도구.

- 앙상블 MLP: 05-4와 같은 MLP 여러 개를 초깃값만 바꿔 한꺼번에 학습한다.
  가중치를 (모델 수, 입력, 출력) 배열로 묶어 행렬곱 한 번으로 모두 계산한다.
- 불확실성: 모델끼리 예측이 얼마나 다른지(표준편차)를 예측값에 대한 %로 나타낸다.
- 후보 풀: 설계 범위를 Sobol 수열로 촘촘히 채운 점들 (04-2와 같은 반올림)
- 넓힌 범위(WIDE): 10장 최적점이 몰린 쪽으로 e/H 0.5, 리브 20개까지 (11-5)
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ch05"))
from data import (INPUTS, LOG, TARGETS, Scaler, from_model,  # noqa: E402,F401
                  load, to_model)

RANGES = {"H": (1.5, 2.5), "e_ratio": (0.2, 0.4), "N": (4, 15),
          "U": (0.05, 0.15)}
WIDE = dict(RANGES, e_ratio=(0.2, 0.5), N=(4, 20))
GAP_MIN = 1.0       # 리브 끝과 가열면 사이 최소 틈 [mm] (10-1)


class EnsembleMLP(nn.Module):
    """SiLU 64x3 MLP (05-4) members개를 묶은 신경망. 출력 (members, n, 3)."""

    def __init__(self, members=5, width=64, depth=3):
        super().__init__()
        dims = [len(INPUTS)] + [width] * depth + [len(TARGETS)]
        self.w, self.b = nn.ParameterList(), nn.ParameterList()
        for a, b in zip(dims[:-1], dims[1:]):
            k = 1 / math.sqrt(a)            # nn.Linear 기본 초기화와 같은 범위
            self.w.append(nn.Parameter(torch.empty(members, a, b)
                                       .uniform_(-k, k)))
            self.b.append(nn.Parameter(torch.empty(members, 1, b)
                                       .uniform_(-k, k)))

    def forward(self, x):
        h = x.expand(len(self.w[0]), -1, -1)
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            h = torch.baddbmm(b, h, w)
            if i < len(self.w) - 1:
                h = F.silu(h)
        return h


class Ensemble:
    """학습한 앙상블. predict(x) -> (평균 예측, 불확실성 %) 원래 단위."""

    def __init__(self, model, sx, sz):
        self.model, self.sx, self.sz = model, sx, sz

    def members(self, x):
        """모델마다의 예측 (members, n, 3), 원래 단위."""
        xs = torch.tensor(self.sx.transform(x), dtype=torch.float32)
        with torch.no_grad():
            z = self.model(xs).numpy().astype(float)
        return np.stack([from_model(self.sz.inverse(m)) for m in z])

    def predict(self, x):
        """평균 예측 (n, 3)과 출력별 불확실성 (n, 3) [%]."""
        y = self.members(x)
        mean = y.mean(axis=0)
        return mean, y.std(axis=0) / mean * 100


def fit_ensemble(x, y, members=5, seed=0, epochs=20000):
    """05-4와 같은 설정(Adam 1e-3, 코사인, full batch)으로 함께 학습한다."""
    sx, sz = Scaler().fit(x), Scaler().fit(to_model(y))
    xt = torch.tensor(sx.transform(x), dtype=torch.float32)
    zt = torch.tensor(sz.transform(to_model(y)), dtype=torch.float32)
    torch.manual_seed(seed)
    model = EnsembleMLP(members)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad()
        # 모델마다 평균제곱오차를 더한다 (서로 다른 가중치라 따로 학습한 것과 같다)
        ((model(xt) - zt) ** 2).mean(dim=(1, 2)).sum().backward()
        opt.step()
        sched.step()
    return Ensemble(model, sx, sz)


def pad(s, width):
    """한글을 두 칸으로 세어 width칸이 되도록 오른쪽을 공백으로 채운다."""
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(width - w, 0)


def summary(u):
    """출력별 불확실성 (n, 3) -> 케이스 하나의 값 (n,): 세 출력 평균."""
    return np.asarray(u).mean(axis=1)


def case_error(y_true, y_pred):
    """케이스마다 세 출력의 상대 오차 [%] 평균 (n,)."""
    return (np.abs(y_pred / y_true - 1) * 100).mean(axis=1)


def to_design(u, ranges=RANGES):
    """0~1 변수 (n, 4) -> 설계 변수 (n, 4). 반올림은 04-2의 lhs()와 같다."""
    (h0, h1), (r0, r1), (n0, n1), (u0, u1) = ranges.values()
    H = np.round(h0 + u[:, 0] * (h1 - h0), 3)
    e = np.round((r0 + u[:, 1] * (r1 - r0)) * H, 3)
    N = np.floor(n0 + u[:, 2] * (n1 - n0 + 1)).clip(n0, n1)
    U = np.round(u0 + u[:, 3] * (u1 - u0), 4)
    return np.column_stack([H, np.round(e / H, 4), N, U])


def to_unit(x, ranges=RANGES):
    """설계 변수 (n, 4) -> 0~1 변수. 거리 계산에 쓴다 (N은 칸의 가운데)."""
    lo = np.array([r[0] for r in ranges.values()], dtype=float)
    hi = np.array([r[1] for r in ranges.values()], dtype=float)
    hi[2] += 1
    u = (np.asarray(x, dtype=float) - lo) / (hi - lo)
    u[:, 2] += 0.5 / (hi[2] - lo[2])
    return u


def pool(n=4096, seed=11, ranges=RANGES):
    """설계 범위를 채우는 후보 n개 (Sobol 수열, 겹치는 점 제거)."""
    from scipy.stats import qmc
    u = qmc.Sobol(d=4, seed=seed).random(n)
    return np.unique(to_design(u, ranges), axis=0)


def outside(x):
    """원래 학습 범위(RANGES) 밖이면 True."""
    return (x[:, 1] > RANGES["e_ratio"][1]) | (x[:, 2] > RANGES["N"][1])


def wide_pool(n=8192, seed=11):
    """넓힌 범위에서 원래 범위 밖이고 틈 제약을 만족하는 후보."""
    x = pool(n, seed, WIDE)
    return x[outside(x) & (x[:, 0] * (1 - x[:, 1]) >= GAP_MIN)]


def corner(x):
    """06-5의 오차가 큰 모서리: e/H > 0.35이고 U > 0.12."""
    return (x[:, 1] > 0.35) & (x[:, 3] > 0.12)


def run_cfd(x, run_dir, workers=6, split="new", log=None):
    """설계 변수 (n, 4)를 04-3의 run_batch.py로 해석해 결과 행 목록으로.

    run_dir에 samples.csv를 쓰고 run_batch.py를 그대로 실행한다. 이미 해석한
    케이스(result.json이 있는 폴더)는 건너뛰므로 다시 실행해도 비용이 없다.
    행마다 status(ok, not_converged, failed)와 dp, dT_max, dT_avg, seconds가 있다.
    """
    import csv
    import subprocess
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "samples.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "split", "H", "e", "N", "U", "e_ratio"])
        for i, (H, er, N, U) in enumerate(x):
            w.writerow([i, split, H, round(H * er, 3), int(N), U, er])
    cmd = [sys.executable, str(ROOT / "ch04" / "run_batch.py"), str(run_dir),
           "--workers", str(workers), "--samples", str(run_dir / "samples.csv")]
    with open(log or run_dir / "run_batch.log", "a") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    with open(run_dir / "results.csv", newline="") as f:
        return list(csv.DictReader(f))


def rows_to_arrays(rows):
    """run_cfd 결과 -> 수렴한 케이스의 (x, y)와 수렴하지 않은 케이스의 x."""
    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]

    def arr(rs, keys):
        return np.array([[float(r[k]) for k in keys] for r in rs]).reshape(
            -1, len(keys))
    return arr(ok, INPUTS), arr(ok, TARGETS), arr(bad, INPUTS)
