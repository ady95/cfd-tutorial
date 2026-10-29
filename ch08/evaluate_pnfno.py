"""책 08-3: PhysicsNeMo FNO와 7장 neuraloperator FNO를 같은 데이터로 비교한다.

1) split마다 물리장 상대 L2 오차 (유체 격자점만)
2) 예측 물리장에서 다시 계산한 dp, dT_max, dT_avg의 상대 오차 (test)
3) 한 건 추론 시간 (GPU), 저장 파일 크기
4) 그림: test 전체의 평균 |dT 오차| 지도를 두 모델 위아래로

사용법: uv run python ch08/evaluate_pnfno.py [PhysicsNeMo 모델] [7장 모델]
    기본값  ~/cfd-runs/ch08/pnfno   ~/cfd-runs/ch07/fno
결과:   표 출력과 ~/cfd-runs/ch08/pnfno-error-map.png
"""
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

from pnfno_model import load_pnfno

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch07"))
from fno_data import load, make_inputs  # noqa: E402
from fno_model import load_fno, predict, rel_l2  # noqa: E402
from grid import scalars  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = ["val", "test", "ood_U", "ood_H"]

pn_path = Path(sys.argv[1] if len(sys.argv) > 1
               else "~/cfd-runs/ch08/pnfno").expanduser()
no_path = Path(sys.argv[2] if len(sys.argv) > 2
               else "~/cfd-runs/ch07/fno").expanduser()
out = pn_path.parent
models = {"neuraloperator": load_fno(no_path)[:2],
          "PhysicsNeMo": load_pnfno(pn_path)[:2]}
data = load()

pred = {name: {} for name in models}
print("물리장 상대 L2 오차 [%] (케이스 평균)")
print("split   모델              Ux      Uy       p      dT")
for s in SPLITS:
    k = data["split"] == s
    for name, (model, scaler) in models.items():
        pred[name][s] = predict(model, scaler, data["mask"][k],
                                data["inputs"][k, 3])
        e = rel_l2(pred[name][s], data["fields"][k], data["mask"][k]).mean(0)
        print(f"{s:7s} {name:15s}" + "".join(f"{v:8.2f}" for v in e))

print("\n물리장에서 다시 계산한 값의 상대 오차 [%] (test 평균)")
print("모델                  dp    dT_max    dT_avg")
k = data["split"] == "test"
cfd = data["cfd_scalars"][k]
for name in models:
    v = np.array([scalars(m, f) for m, f in
                  zip(data["mask"][k], pred[name]["test"])])
    print(f"{name:15s}" + "".join(f"{x:10.3f}" for x in
                                  np.abs(v / cfd - 1).mean(0) * 100))

print("\n한 건 추론 (GPU)과 저장 파일")
x1 = torch.as_tensor(make_inputs(data["mask"][:1], data["inputs"][:1, 3]),
                     device="cuda")
files = {"neuraloperator": [no_path.with_suffix(".pt"),
                            no_path.with_suffix(".json")],
         "PhysicsNeMo": [pn_path.with_suffix(".mdlus"),
                         pn_path.with_suffix(".json")]}
for name, (model, _) in models.items():
    with torch.no_grad():
        for _ in range(10):
            model(x1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            model(x1)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / 100 * 1e3
    size = sum(f.stat().st_size for f in files[name]) / 1e6
    names = " + ".join(f.name for f in files[name])
    print(f"{name:15s} {ms:6.2f} ms   {size:5.1f} MB ({names})")

# 그림: test 전체의 평균 |dT 오차|. 색 범위를 두 그림이 같게 맞춘다
mk = data["mask"][k].astype(bool)
cnt = mk.sum(0)
maps = {}
for name in models:
    e = np.abs(pred[name]["test"][:, 3] - data["fields"][k][:, 3])
    maps[name] = np.where(cnt > 0, (e * mk).sum(0) / np.maximum(cnt, 1),
                          np.nan)
vmax = max(np.nanpercentile(a, 99) for a in maps.values())
fig, axes = plt.subplots(2, 1, figsize=(14, 4.6), sharex=True)
for ax, (name, a) in zip(axes, maps.items()):
    im = ax.imshow(a, origin="lower", cmap="magma", aspect="auto",
                   extent=(0, 80, 0, 3), vmin=0, vmax=vmax)
    ax.set_title(f"{name}: test mean |dT error| (K)", fontsize=10)
    ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, shrink=0.9)
axes[-1].set_xlabel("x (mm)")
fig.tight_layout()
fig.savefig(out / "pnfno-error-map.png", dpi=130)
print("저장:", out / "pnfno-error-map.png")
