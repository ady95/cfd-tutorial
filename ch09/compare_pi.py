"""책 09-4·09-5: 물리 손실 적용 전후의 FNO를 같은 데이터로 비교한다.

모델마다 split별로
- 물리장 상대 L2 오차 (Ux, Uy, p, dT), 07-5와 같음
- 보존 잔차: 단면 평균 |질량 잔차|, |에너지 잔차| (09-2와 같음)
- (test) 물리장에서 다시 계산한 dp, dT_max, dT_avg의 상대 오차
를 계산한다. 모델은 이름=경로 형식으로 준다 (경로는 .pt와 .json의 공통 이름).

사용법: uv run python ch09/compare_pi.py [이름=경로 ...] [--splits test ood_H]
    기본값  7장=~/cfd-runs/ch07/fno  λ0.1=~/cfd-runs/ch09/full/lam01
           λ1=~/cfd-runs/ch09/full/lam1  λ1+넓은입력=~/cfd-runs/ch09/full/lam1_wide
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from physics import all_weights, residuals

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch07"))
from fno_data import load  # noqa: E402
from fno_model import load_fno, predict, rel_l2  # noqa: E402
from grid import scalars  # noqa: E402

DEFAULT = ["7장=~/cfd-runs/ch07/fno", "λ0.1=~/cfd-runs/ch09/full/lam01",
           "λ1=~/cfd-runs/ch09/full/lam1",
           "λ1+넓은입력=~/cfd-runs/ch09/full/lam1_wide"]
p = argparse.ArgumentParser()
p.add_argument("models", nargs="*", default=DEFAULT)
p.add_argument("--splits", nargs="+", default=["val", "test", "ood_U", "ood_H"])
args = p.parse_args()


def width(s):
    return sum(2 if ord(c) > 0x1100 else 1 for c in s)   # 한글은 2칸


def pad(s, n):
    return s + " " * (n - width(s))


data = load()
models = [(m.split("=")[0],
           load_fno(Path(m.split("=")[1]).expanduser())[:2])
          for m in args.models]
phys = {s: all_weights(data, data["split"] == s) for s in args.splits}

print("물리장 상대 L2 [%]과 보존 잔차 [%] (케이스 평균)")
print("split  모델            Ux     Uy      p     dT    질량  에너지")
scal = {}
for s in args.splits:
    k = data["split"] == s
    for name, (model, scaler) in models:
        pred = predict(model, scaler, data["mask"][k], data["inputs"][k, 3])
        e = rel_l2(pred, data["fields"][k], data["mask"][k]).mean(0)
        rm, re = residuals(pred[:, 0], pred[:, 3], *phys[s])
        print(f"{s:6s} {pad(name, 13)}" + "".join(f"{v:7.2f}" for v in e)
              + f"{np.abs(rm).mean() * 100:7.2f}{np.abs(re).mean() * 100:7.2f}")
        if s == "test":
            v = np.array([scalars(m, f)
                          for m, f in zip(data["mask"][k], pred)])
            scal[name] = np.abs(v / data["cfd_scalars"][k] - 1).mean(0) * 100

if scal:
    print("\n물리장에서 다시 계산한 값의 상대 오차 [%] (test 평균)")
    print("모델                 dp  dT_max  dT_avg")
    for name, v in scal.items():
        print(pad(name, 17) + "".join(f"{x:8.3f}" for x in v))
