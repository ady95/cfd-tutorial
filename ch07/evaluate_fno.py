"""책 07-5: FNO 예측 물리장을 CFD와 비교하고, 5장 MLP와 견준다.

1) split마다 물리장 상대 L2 오차 (유체 격자점만)
2) 예측 물리장에서 dp, dT_max, dT_avg를 다시 계산해 CFD 값과 비교.
   같은 계산을 CFD 격자 배열에 한 값(옮기기 오차)과 최종 MLP도 함께 본다
3) 한 건 추론 시간 (GPU)
4) 그림: test 케이스 하나의 CFD·FNO·오차, test 전체의 평균 오차 지도

사용법: uv run python ch07/evaluate_fno.py [모델] [출력 폴더]
    기본값  ~/cfd-runs/ch07/fno   ~/cfd-runs/ch07
"""
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

from fno_data import load, make_inputs
from fno_model import load_fno, predict, rel_l2
from grid import scalars

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch06"))
from common import load_mlp  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = ["val", "test", "ood_U", "ood_H"]

model_path = Path(sys.argv[1] if len(sys.argv) > 1
                  else "~/cfd-runs/ch07/fno").expanduser()
out = Path(sys.argv[2] if len(sys.argv) > 2
           else "~/cfd-runs/ch07").expanduser()
out.mkdir(parents=True, exist_ok=True)
model, scaler, meta = load_fno(model_path)
data = load()
mlp = load_mlp()

pred, err_l2 = {}, {}
print("물리장 상대 L2 오차 [%] (케이스 평균 / 최대)")
print("split      개수        Ux           Uy            p           dT")
for s in SPLITS:
    k = data["split"] == s
    pred[s] = predict(model, scaler, data["mask"][k], data["inputs"][k, 3])
    err_l2[s] = rel_l2(pred[s], data["fields"][k], data["mask"][k])
    print(f"{s:7s} {k.sum():6d}  " + "  ".join(
        f"{m:5.2f}/{x:5.2f}" for m, x in
        zip(err_l2[s].mean(0), err_l2[s].max(0))))

print("\n물리장에서 다시 계산한 값의 상대 오차 [%] (CFD 함수 객체 값 기준, 평균)")
print("split    대상                 dp    dT_max    dT_avg")
for s in SPLITS[1:]:
    k = data["split"] == s
    cfd = data["cfd_scalars"][k]
    fno = np.array([scalars(m, f) for m, f in zip(data["mask"][k], pred[s])])
    rows = [("CFD 격자 배열", data["grid_scalars"][k]), ("FNO 물리장", fno),
            ("MLP (5장)", mlp(data["inputs"][k].astype(float)))]
    for name, v in rows:
        e = np.abs(v / cfd - 1).mean(0) * 100
        print(f"{s:7s}  {name:<14s}" + "".join(f"{x:9.3f}" for x in e))

# 한 건 추론 시간
x1 = torch.as_tensor(make_inputs(data["mask"][:1], data["inputs"][:1, 3]),
                     device="cuda")
with torch.no_grad():
    for _ in range(10):
        model(x1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        model(x1)
    torch.cuda.synchronize()
one = (time.perf_counter() - start) / 100
k = data["split"] == "test"
start = time.perf_counter()
predict(model, scaler, data["mask"][k], data["inputs"][k, 3])
batch = (time.perf_counter() - start) / k.sum()
print(f"\nFNO 추론 (GPU): 1건 {one * 1e3:.2f} ms, "
      f"test {k.sum()}건 묶음 {batch * 1e3:.2f} ms/건 (CPU로 옮기기 포함)")

# 그림 1: dT 오차가 중앙값인 test 케이스
idx = np.flatnonzero(k)
j = np.argsort(err_l2["test"][:, 3])[len(idx) // 2]
i = idx[j]
m = data["mask"][i].astype(bool)
nx = m.shape[1]
x0, x1 = np.searchsorted(np.linspace(0, 80, nx), [40, 72])
fig, axes = plt.subplots(3, 2, figsize=(14, 6.2), sharex=True, sharey=True)
for col, (c, name, cmap) in enumerate([(3, "dT [K]", "inferno"),
                                       (0, "Ux [m/s]", "viridis")]):
    t, p = data["fields"][i, c], pred["test"][j, c]
    for row, (title, g) in enumerate([("CFD", t), ("FNO", p)]):
        ax = axes[row, col]
        im = ax.imshow(np.where(m, g, np.nan)[:, x0:x1], origin="lower",
                       cmap=cmap, aspect="auto", extent=(40, 72, 0, 3),
                       vmin=t[m].min(), vmax=t[m].max())
        ax.set_title(f"{name}: {title}", fontsize=10)
    fig.colorbar(im, ax=axes[:2, col], shrink=0.8)
    ax = axes[2, col]
    d = np.where(m, p - t, np.nan)[:, x0:x1]
    lim = np.nanmax(np.abs(d))
    im = ax.imshow(d, origin="lower", cmap="RdBu_r", aspect="auto",
                   extent=(40, 72, 0, 3), vmin=-lim, vmax=lim)
    ax.set_title(f"{name}: FNO - CFD", fontsize=10)
    ax.set_xlabel("x [mm]")
    fig.colorbar(im, ax=ax, shrink=0.8)
for ax in axes[:, 0]:
    ax.set_ylabel("y [mm]")
fig.suptitle(f"test case {data['case'][i]} (median dT error "
             f"{err_l2['test'][j, 3]:.2f}%)", fontsize=10)
fig.savefig(out / "fno-case.png", dpi=130)

# 그림 2: test 전체의 평균 |오차| 지도 (그 자리가 유체인 케이스만 평균)
mk = data["mask"][k].astype(bool)
fig, axes = plt.subplots(2, 1, figsize=(14, 4.6), sharex=True)
for ax, (c, name) in zip(axes, [(3, "|dT error| [K]"),
                                (0, "|Ux error| [m/s]")]):
    e = np.abs(pred["test"][:, c] - data["fields"][k][:, c])
    cnt = mk.sum(0)
    avg = np.where(cnt > 0, (e * mk).sum(0) / np.maximum(cnt, 1), np.nan)
    # 가열 시작·끝 점의 몇 칸이 유난히 커서, 색 범위는 상위 1%에서 자른다
    im = ax.imshow(avg, origin="lower", cmap="magma", aspect="auto",
                   extent=(0, 80, 0, 3), vmax=np.nanpercentile(avg, 99))
    ax.set_title(f"test mean {name}", fontsize=10)
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, shrink=0.9)
axes[-1].set_xlabel("x [mm]")
fig.tight_layout()
fig.savefig(out / "fno-error-map.png", dpi=130)
print("저장:", out / "fno-case.png", out / "fno-error-map.png")
