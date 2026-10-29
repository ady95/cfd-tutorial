"""책 09-5: 손실 항마다 모델 파라미터를 얼마나 세게 당기는지(기울기 크기) 잰다.

학습한 모델에 train 묶음을 넣고 데이터 손실, 질량 손실, 에너지 손실을 각각
역전파해 파라미터 기울기의 노름을 구한다. 손실 값이 비슷해도 기울기가 몇 배
크면, λ = 1에서도 그 항이 학습을 끌고 간다.

사용법: uv run python ch09/grad_balance.py [이름=경로 ...]
    기본값  7장=~/cfd-runs/ch07/fno  λ0.1=~/cfd-runs/ch09/full/lam01
           λ1=~/cfd-runs/ch09/full/lam1
"""
import sys
from pathlib import Path

import numpy as np
import torch

from physics import all_weights, residuals

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch07"))
from fno_data import U_REF, load, split_tensors  # noqa: E402
from fno_model import load_fno  # noqa: E402

DEFAULT = ["7장=~/cfd-runs/ch07/fno", "λ0.1=~/cfd-runs/ch09/full/lam01",
           "λ1=~/cfd-runs/ch09/full/lam1"]
BATCHES = 20


def width(s):
    return sum(2 if ord(c) > 0x1100 else 1 for c in s)   # 한글은 2칸


def losses(model, x, y, m, W, q, e, u, rms):
    z = model(x)
    mm = m[:, None]
    num = (((z - y) * mm) ** 2).sum(dim=(2, 3)).sqrt()
    l_data = (num / ((y * mm) ** 2).sum(dim=(2, 3)).sqrt()).mean()
    rm, re = residuals(z[:, 0] * rms[0] * u[:, None, None], z[:, 3] * rms[3],
                       W, q, e)
    return [l_data, rm.pow(2).mean(1).sqrt().mean(),
            re.pow(2).mean(1).sqrt().mean()]


def grad_norm(model, loss):
    model.zero_grad()
    loss.backward(retain_graph=True)
    # 스펙트럼 가중치는 복소수라 절댓값의 제곱을 더한다
    return torch.sqrt(sum((p.grad.abs() ** 2).sum()
                          for p in model.parameters()
                          if p.grad is not None)).item()


data = load()
k = data["split"] == "train"
W, q, e = all_weights(data, k)
u = data["inputs"][k, 3] / U_REF
P = [torch.as_tensor(a, device="cuda") for a in (W, q, e, u)]
names = sys.argv[1:] or DEFAULT

print(f"train 묶음 {BATCHES}개(8개씩) 평균. 기울기는 파라미터 전체의 L2 노름")
print("모델          손실: 데이터   질량  에너지 | 기울기: 데이터    질량  에너지")
for item in names:
    name, path = item.split("=")
    model, scaler, _ = load_fno(Path(path).expanduser())
    model.train()
    x, y, m = split_tensors(data, "train", scaler)
    rms = torch.as_tensor(scaler.rms, device="cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    order = torch.randperm(len(x), generator=g)[:BATCHES * 8]
    L, G = [], []
    for i in range(BATCHES):
        b = order[i * 8:(i + 1) * 8].cuda()
        ls = losses(model, x[b], y[b], m[b], *[t[b] for t in P], rms)
        L.append([v.item() for v in ls])
        G.append([grad_norm(model, v) for v in ls])
    L, G = np.mean(L, 0), np.mean(G, 0)
    print(name + " " * (14 - width(name)) + "      "
          + "".join(f"{v:7.4f}" for v in L) + " |        "
          + "".join(f"{v:7.2f}" for v in G))
