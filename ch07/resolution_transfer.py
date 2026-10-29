"""책 07-5: 학습한 해상도와 다른 격자에서 FNO를 실행해 본다 (07-1 해상도 독립성).

1025 x 49로 학습한 모델과 513 x 25(한 칸씩 건너뛴 격자)로 학습한 모델을
test 90개에서 두 해상도로 각각 실행해, 물리장 오차와 다시 계산한 값의
오차를 비교한다.

사용법: uv run python ch07/resolution_transfer.py [모델] [절반 해상도 모델]
    기본값  ~/cfd-runs/ch07/fno   ~/cfd-runs/ch07/fno_half
"""
import sys
from pathlib import Path

import numpy as np

from fno_data import load
from fno_model import load_fno, predict, rel_l2
from grid import scalars

full = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/ch07/fno")
half = Path(sys.argv[2] if len(sys.argv) > 2 else "~/cfd-runs/ch07/fno_half")
data = load()
k = data["split"] == "test"
cfd = data["cfd_scalars"][k]
print("학습 격자   실행 격자   물리장 상대 L2 [%]" + " " * 11 + "값 상대 오차 [%]")
print(" " * 24 + " ".join(f"{n:>5s}" for n in ["Ux", "Uy", "p", "dT"])
      + "      " + " ".join(f"{n:>6s}" for n in ["dp", "dT_max", "dT_avg"]))
for path in (full.expanduser(), half.expanduser()):
    model, scaler, meta = load_fno(path)
    for step in (1, 2):
        mask = data["mask"][k][:, ::step, ::step]
        true = data["fields"][k][:, :, ::step, ::step]
        pred = predict(model, scaler, mask, data["inputs"][k, 3])
        e = rel_l2(pred, true, mask).mean(0)
        s = np.array([scalars(m, f) for m, f in zip(mask, pred)])
        es = np.abs(s / cfd - 1).mean(0) * 100
        trained, run = (f"{1024 // meta['step'] + 1} x "
                        f"{48 // meta['step'] + 1}",
                        f"{mask.shape[2]} x {mask.shape[1]}")
        print(f"{trained:10s}  {run:10s}  " + " ".join(f"{v:5.2f}" for v in e)
              + "      " + " ".join(f"{v:6.3f}" for v in es))
