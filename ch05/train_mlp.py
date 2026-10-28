"""책 05-4: PyTorch MLP로 세 출력을 함께 예측하는 대리모델을 학습한다.

1) GPU를 확인하고, 기본 MLP(은닉층 3개 x 64, SiLU)를 학습하며 손실 곡선을 남긴다.
2) 활성화 함수(ReLU, Tanh)와 난수 시드를 바꿔 결과가 얼마나 달라지는지 본다.
3) CPU와 GPU의 학습 속도를 비교한다.
4) 05-2·05-3의 모델과 val 상대 오차 분포를 비교하는 그림을 그린다.

사용법: uv run python ch05/train_mlp.py [출력 폴더]   (기본 ~/cfd-runs/ch05)
결과:   표 출력과 출력 폴더의 mlp-training.png
"""
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch import nn

from data import TARGETS, Scaler, from_model, load, rel_error, to_model

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

EPOCHS, LR = 20000, 1e-3

x_tr, y_tr, _ = load("train")
x_va, y_va, _ = load("val")
sx = Scaler().fit(x_tr)                     # 입력 표준화 (train으로만)
sz = Scaler().fit(to_model(y_tr))           # 출력도 로그 변환 뒤 표준화


def mlp(act=nn.SiLU, width=64, depth=3):
    layers, n = [], x_tr.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(n, width), act()]
        n = width
    return nn.Sequential(*layers, nn.Linear(n, len(TARGETS)))


def tensor(a, device):
    return torch.tensor(a, dtype=torch.float32, device=device)


def train(act=nn.SiLU, seed=0, epochs=EPOCHS, device="cpu", log=None):
    """train 전체를 한 번에 넣어(full batch) epochs번 학습한다."""
    torch.manual_seed(seed)
    model = mlp(act).to(device)
    x, z = tensor(sx.transform(x_tr), device), tensor(
        sz.transform(to_model(y_tr)), device)
    xv, zv = tensor(sx.transform(x_va), device), tensor(
        sz.transform(to_model(y_va)), device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    start = time.perf_counter()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = ((model(x) - z) ** 2).mean()     # 평균 제곱 오차
        loss.backward()
        opt.step()
        sched.step()                            # 학습률을 코사인 모양으로 줄임
        if log is not None and ep % 100 == 0:
            with torch.no_grad():
                log.append((ep, loss.item(),
                            ((model(xv) - zv) ** 2).mean().item()))
    if device == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - start
    return model.cpu(), sec


def predict(model, x):
    with torch.no_grad():
        z = model(tensor(sx.transform(x), "cpu")).numpy().astype(float)
    return from_model(sz.inverse(z))


def row(name, y, p, sec=None):
    e = rel_error(y, p)
    pad = 12 - sum(2 if ord(c) > 0x1100 else 1 for c in name)  # 한글은 2칸
    t = f"{sec:5.1f}초" if sec is not None else " " * 7
    print(name + " " * pad + t + "".join(
        f"  {e[k][0]:5.2f}% {e[k][1]:4.1f}%" for k in TARGETS))


# 1) GPU 확인과 기본 모델 학습
print("PyTorch", torch.__version__, "| CUDA",
      torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
n_param = sum(p.numel() for p in mlp().parameters())
print(f"MLP 가중치 {n_param}개\n")
print(" " * 19 + "".join(f"{t:>14}" for t in TARGETS))
print("모델        시간" + "    평균  최대" * len(TARGETS))
log = []
base, sec = train(log=log)
row("SiLU (train)", y_tr, predict(base, x_tr))
row("SiLU", y_va, predict(base, x_va), sec)

# 2) 활성화 함수와 난수 시드
for act in [nn.ReLU, nn.Tanh]:
    m, sec = train(act)
    row(act.__name__, y_va, predict(m, x_va), sec)
for seed in [1, 2]:
    m, sec = train(seed=seed)
    row(f"SiLU 시드 {seed}", y_va, predict(m, x_va), sec)

# 3) CPU와 GPU 속도: 2000 에포크
if torch.cuda.is_available():
    for device in ["cpu", "cuda"]:
        _, sec = train(epochs=2000, device=device)
        print(f"{device:<5} 2000 에포크 {sec:5.1f}초 ({sec / 2:.2f} ms/에포크)")

# 4) 그림: (a) 손실 곡선, (b) 모델별 val 상대 오차 분포
z_tr = to_model(y_tr)
others = {}
for d in (3, 6):
    p = make_pipeline(StandardScaler(), PolynomialFeatures(d),
                      LinearRegression()).fit(x_tr, z_tr)
    others[f"poly {d}"] = from_model(p.predict(x_va))
params = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.05,
          "subsample": 0.8, "seed": 0}
rounds = [1819, 2393, 1718]          # 05-3에서 교차 검증으로 정한 트리 수
others["XGBoost"] = from_model(np.column_stack([
    xgb.train(params, xgb.DMatrix(x_tr, label=z_tr[:, i]), n).predict(
        xgb.DMatrix(x_va)) for i, n in enumerate(rounds)]))
others["MLP"] = predict(base, x_va)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/ch05")
out = out.expanduser()
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 4, figsize=(16, 4),
                         gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
ep, tr, va = np.array(log).T
axes[0].semilogy(ep, tr, color="#2a78d6", lw=2, label="train")
axes[0].semilogy(ep, va, color="#eb6834", lw=2, label="val")
axes[0].set(xlabel="epoch", ylabel="mean squared error (scaled)",
            title="(a) loss curve")
axes[0].set_xticks(range(0, EPOCHS + 1, 5000),
                   [f"{k // 1000}k" if k else "0"
                    for k in range(0, EPOCHS + 1, 5000)])
axes[0].legend()
for i, (ax, t) in enumerate(zip(axes[1:], TARGETS)):
    err = [np.abs(p[:, i] / y_va[:, i] - 1) * 100 for p in others.values()]
    ax.boxplot(err, tick_labels=list(others), showfliers=True,
               flierprops={"markersize": 3})
    ax.set(yscale="log", title=f"({'bcd'[i]}) {t} on val",
           ylabel="relative error [%]" if i == 0 else "")
for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "mlp-training.png", dpi=150)
