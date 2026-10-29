"""책 08-2: 공식 darcy_fno 예제가 저장한 모델을 불러와 새 표본으로 확인한다.

PhysicsNeMo 컨테이너 안에서 darcy_fno 폴더를 작업 폴더로 두고 실행한다.
1) checkpoints/의 마지막 .mdlus를 Module.from_checkpoint로 불러온다
   (모델 클래스와 생성 인자를 적지 않는다)
2) Darcy2D로 학습에 쓰지 않은 표본 256개를 새로 만들어 상대 L2 오차를 잰다
3) 표본 하나의 투과율·정답·예측·오차를 darcy-check.png로 저장한다

사용법 (컨테이너 안): python /tutorial/ch08/check_darcy.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import torch

import physicsnemo
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NORM = {"permeability": (1.25, 0.75), "darcy": (4.52e-2, 2.79e-2)}  # config.yaml

ckpt = sorted(Path("checkpoints").glob("FNO.0.*.mdlus"),
              key=lambda p: int(p.stem.split(".")[-1]))[-1]
model = physicsnemo.Module.from_checkpoint(str(ckpt)).cuda().eval()
print(f"불러온 파일: {ckpt}  ({type(model).__name__}, "
      f"파라미터 {sum(p.numel() for p in model.parameters()):,}개)")

np.random.seed(1)       # Darcy2D는 투과율을 numpy 난수로 만든다
data = Darcy2D(resolution=256, batch_size=64, normaliser=NORM)
mean, std = NORM["darcy"]
errs = []
with torch.no_grad():
    for _, batch in zip(range(4), data):
        pred = model(batch["permeability"]) * std + mean
        true = batch["darcy"] * std + mean
        num = ((pred - true) ** 2).sum(dim=(1, 2, 3)).sqrt()
        errs.append(num / (true ** 2).sum(dim=(1, 2, 3)).sqrt())
err = torch.cat(errs) * 100
print(f"새 표본 {len(err)}개의 상대 L2 오차: 평균 {err.mean():.2f}%, "
      f"중앙값 {err.median():.2f}%, 최대 {err.max():.2f}%")

k = batch["permeability"][0, 0].cpu() * NORM["permeability"][1] \
    + NORM["permeability"][0]
t, p = true[0, 0].cpu(), pred[0, 0].cpu()
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, (title, img, cmap) in zip(axes, [
        ("permeability k", k, "gray"), ("truth u", t, "viridis"),
        ("FNO u", p, "viridis"), ("FNO - truth", p - t, "RdBu_r")]):
    lim = float((p - t).abs().max())
    kw = dict(vmin=-lim, vmax=lim) if cmap == "RdBu_r" else {}
    im = ax.imshow(img, cmap=cmap, **kw)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("darcy-check.png", dpi=120)
print("저장: darcy-check.png")
