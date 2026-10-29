"""책 08-4: 숫자 모델(XGBoost, MLP)과 물리장 모델(FNO 두 가지)을 한 표로 비교한다.

모든 모델에서 같은 세 값(dp, dT_max, dT_avg)을 CFD 함수 객체 값과 비교한다.
- XGBoost, MLP: 06-5와 같이 train으로 새로 학습한다 (CPU)
- FNO: 7장·08-3에서 학습해 저장한 모델로 물리장을 예측하고, 그 물리장에서
  세 값을 다시 계산한다. 학습 시간은 저장할 때 적은 값 (GPU)
추론 1건은 입력 하나에서 세 값을 얻기까지의 시간이다. FNO는 GPU 예측,
CPU로 옮기기, 물리장에서 세 값 계산까지 포함한다.

사용법: uv run python ch08/compare_all.py      (약 1분, MLP 학습 포함)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pnfno_model import load_pnfno

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ch07"))
sys.path.insert(0, str(ROOT / "ch06"))
from common import fit_mlp, fit_xgb, load  # noqa: E402
from fno_data import load as load_fields  # noqa: E402
from fno_model import load_fno, predict  # noqa: E402
from grid import scalars  # noqa: E402

SPLITS = ["test", "ood_U", "ood_H"]
torch.set_num_threads(6)


def width(s):
    return sum(2 if ord(c) > 0x1100 else 1 for c in s)   # 한글은 2칸


def rel_err(y, p):
    """케이스 x 출력 상대 오차 [%]."""
    return np.abs(p / y - 1) * 100


def timed(f, n=50):
    for _ in range(5):
        f()
    start = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - start) / n


x_tr, y_tr, _ = load("train")
sets = {s: load(s)[:2] for s in SPLITS}
fields = load_fields()
for s in SPLITS:                     # 두 데이터에서 같은 케이스인지 확인
    k = fields["split"] == s
    assert np.allclose(fields["cfd_scalars"][k], sets[s][1])

rows = []
for name, fit in [("XGBoost", fit_xgb), ("MLP", fit_mlp)]:
    start = time.perf_counter()
    f = fit(x_tr, y_tr)
    sec = time.perf_counter() - start
    err = {s: rel_err(y, f(x)) for s, (x, y) in sets.items()}
    rows.append((name, "CPU", sec, err, timed(lambda: f(x_tr[:1]))))

for name, (model, scaler, meta) in [
        ("FNO neuralop", load_fno(Path("~/cfd-runs/ch07/fno").expanduser())),
        ("FNO PhysicsNeMo", load_pnfno("~/cfd-runs/ch08/pnfno"))]:
    err = {}
    for s in SPLITS:
        k = fields["split"] == s
        mask, U = fields["mask"][k], fields["inputs"][k, 3]
        pred = predict(model, scaler, mask, U)
        v = np.array([scalars(m, p) for m, p in zip(mask, pred)])
        err[s] = rel_err(fields["cfd_scalars"][k], v)
    m1, u1 = fields["mask"][:1], fields["inputs"][:1, 3]

    def one():
        return scalars(m1[0], predict(model, scaler, m1, u1)[0])
    rows.append((name, "GPU", meta["train_seconds"], err, timed(one)))

print("세 값(dp, dT_max, dT_avg) 상대 오차 [%]. test 최대 말고는 케이스 평균")
print("모델             학습 시간    test test 최대    ood_U    ood_H  추론 1건")
for name, dev, sec, err, t in rows:
    print(name + " " * (17 - width(name)) + f"{sec:7.1f}초"
          f"{err['test'].mean():8.3f}{err['test'].max():10.2f}"
          f"{err['ood_U'].mean():9.3f}{err['ood_H'].mean():9.3f}"
          f"{t * 1e3:8.2f}ms")

print("\n출력별 test 평균 상대 오차 [%]")
print("모델                  dp    dT_max    dT_avg")
for name, _, _, err, _ in rows:
    print(name + " " * (17 - width(name))
          + "".join(f"{v:10.3f}" for v in err["test"].mean(0)))
