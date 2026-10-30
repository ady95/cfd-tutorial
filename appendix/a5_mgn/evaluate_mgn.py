"""책 부록 A-5: MeshGraphNet 예측을 CFD와 비교한다 (07-5의 FNO 평가와 같은 순서).

1) split마다 셀 값 물리장의 상대 L2 오차
2) 예측 셀 값에서 dp, dT_max, dT_avg를 다시 계산해 데이터셋 값과 비교
3) 그래프 하나 추론 시간 (GPU)
4) 그림: dT 오차가 중앙값인 test 케이스의 CFD·MGN·오차

사용법 (PhysicsNeMo 컨테이너 안):
    python /tutorial/appendix/a5_mgn/evaluate_mgn.py /out/graphs.pt /out
"""
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

from physicsnemo.models.meshgraphnet import MeshGraphNet

from mgn_common import denorm, rel_l2, scalars, to_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = ["val", "test", "ood_U", "ood_H"]

graphs = torch.load(sys.argv[1], weights_only=False)
out = Path(sys.argv[2])
ck = torch.load(out / "mgn.pt", weights_only=False)
rms = ck["rms"]
dev = "cuda"
model = MeshGraphNet(input_dim_nodes=ck["dims"][0],
                     input_dim_edges=ck["dims"][1], output_dim=4).to(dev)
model.load_state_dict(ck["model"])
model.eval()


def run(g):
    d = to_data(g, rms, dev)
    with torch.no_grad():
        return denorm(model(d.x, d.edge_attr, d), g, rms)


preds, l2 = {}, {}
print("셀 값 물리장 상대 L2 오차 [%] (케이스 평균 / 최대)")
print("split      개수        Ux           Uy            p           dT")
for s in SPLITS:
    gs = [g for g in graphs if g["split"] == s]
    preds[s] = [run(g) for g in gs]
    l2[s] = np.array([rel_l2(p, g["y"]) for p, g in zip(preds[s], gs)])
    print(f"{s:7s} {len(gs):6d}  " + "  ".join(
        f"{m:5.2f}/{x:5.2f}" for m, x in zip(l2[s].mean(0), l2[s].max(0))))

print("\n셀 값에서 다시 계산한 값의 상대 오차 [%] (데이터셋 값 기준, 평균/최대)")
print("split          dp            dT_max        dT_avg")
for s in SPLITS[1:]:
    gs = [g for g in graphs if g["split"] == s]
    e = np.array([np.abs(scalars(p, g) / np.array(g["cfd"]) - 1)
                  for p, g in zip(preds[s], gs)]) * 100
    print(f"{s:7s}" + "".join(f"  {m:6.3f}/{x:6.3f}"
                              for m, x in zip(e.mean(0), e.max(0))))

# 추론 시간: 노드 수가 중앙값인 test 그래프 하나
test = [g for g in graphs if g["split"] == "test"]
g = sorted(test, key=lambda g: len(g["x"]))[len(test) // 2]
d = to_data(g, rms, dev)
with torch.no_grad():
    for _ in range(5):
        model(d.x, d.edge_attr, d)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        model(d.x, d.edge_attr, d)
    torch.cuda.synchronize()
print(f"\n그래프 하나 추론 (GPU, 노드 {len(g['x']):,}개): "
      f"{(time.perf_counter() - start) / 20 * 1e3:.1f} ms")
print(f"학습 시간 {ck['seconds']:.0f}초")

# 그림: dT 오차가 중앙값인 test 케이스, x 40~72 mm
j = int(np.argsort(l2["test"][:, 3])[len(test) // 2])
g, p = test[j], preds["test"][j].numpy()
xy = g["x"][:, :2].numpy() * [80, 3]
true = g["y"][:, 3].numpy()
keep = (xy[:, 0] >= 40) & (xy[:, 0] <= 72)
fig, axes = plt.subplots(3, 1, figsize=(14, 6.6), sharex=True)
vmax = true[keep].max()
for ax, v, title, cmap, lim in [
        (axes[0], true, "CFD dT (K)", "inferno", (0, vmax)),
        (axes[1], p[:, 3], "MeshGraphNet dT (K)", "inferno", (0, vmax)),
        (axes[2], p[:, 3] - true, "error (K)", "RdBu_r", None)]:
    if lim is None:
        a = np.abs(v[keep]).max()
        lim = (-a, a)
    sc = ax.scatter(xy[keep, 0], xy[keep, 1], c=v[keep], s=1.2, marker="s",
                    cmap=cmap, vmin=lim[0], vmax=lim[1], linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("y (mm)")
    ax.set_aspect(2)
    fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)
axes[-1].set_xlabel("x (mm)")
fig.suptitle(f"test case {g['case']} (median dT error "
             f"{l2['test'][j, 3]:.2f}%)", fontsize=12)
fig.tight_layout()
fig.savefig(out / "mgn-case.png", dpi=130)
print("저장:", out / "mgn-case.png")
