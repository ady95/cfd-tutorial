"""책 10-5: 최적 설계 후보를 CFD로 다시 해석해 대리모델 예측과 비교한다.

pareto.py가 저장한 후보 A, B, C와 기준 설계를 높이 방향 셀 30개(데이터셋과
같은 격자)와 60개(03-5의 촘촘한 격자)로 해석한다.
- 예측과 CFD(셀 30)의 차이: 대리모델 오차
- 셀 30과 셀 60의 차이: 격자 오차
- 기준 설계 대비 개선량: 예측, CFD 셀 30, CFD 셀 60 세 가지로

사용법: uv run python ch10/verify.py      (OpenFOAM 필요, 약 10분)
결과:   표 출력, CFD 작업 폴더 ~/cfd-runs/ch10/verify/nh30, nh60
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from problem import BASE, NAMES, base_values, cfd

OUT = Path("~/cfd-runs/ch10").expanduser()

if shutil.which("foamRun") is None:
    sys.exit("OpenFOAM 환경이 없습니다: . /opt/openfoam14/etc/bashrc")
cands = json.loads((OUT / "candidates.json").read_text())
names = ["기준"] + list(cands)
x = np.array([BASE] + [[c[n] for n in NAMES] for c in cands.values()])
pred = [base_values()] + [c["pred"] for c in cands.values()]
res = {nh: cfd(x, OUT / "verify" / f"nh{nh}", cells_h=nh) for nh in (30, 60)}

print("설계   격자  상태          반복    시간")
for nh, rs in res.items():
    for name, r in zip(names, rs):
        print(f"{name:<4}   셀 {nh}  {r['status']:<13} {r['iterations']:>5}"
              f"  {r['seconds']:5.1f}초")

for key, label in [("dT_max", "최고 온도 상승 [K]"), ("P", "펌핑 동력 [mW/m]")]:
    print(f"\n{label}")
    print("설계     예측    셀 30    셀 60   예측 오차  격자 오차")
    for k, name in enumerate(names):
        p, c30, c60 = pred[k][key], res[30][k][key], res[60][k][key]
        print(f"{name:<4} {p:8.3f} {c30:8.3f} {c60:8.3f}"
              f"   {(p / c30 - 1) * 100:+6.2f}%   {(c30 / c60 - 1) * 100:+6.2f}%")

print("\n기준 설계 대비 변화 (dT_max [K], P [%])")
print("설계        예측            셀 30           셀 60")
for k, name in list(enumerate(names))[1:]:
    cells = []
    for src in (pred, res[30], res[60]):
        dT = src[k]["dT_max"] - src[0]["dT_max"]
        dP = (src[k]["P"] / src[0]["P"] - 1) * 100
        cells.append(f"{dT:+6.3f} {dP:+6.1f}%")
    print(f"{name:<4}  " + "   ".join(cells))
