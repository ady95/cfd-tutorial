"""책 10-1: 설계 변수와 제약조건이 설계 공간을 얼마나 줄이는지 본다.

1) 기준 설계의 대리모델 예측
2) 제약마다 격자(101 x 81 x 12 x 101)에서 남는 설계의 비율
3) 학습 범위 밖으로 나가면 대리모델이 무엇을 내놓는지 (신뢰 범위 제약의 이유)
   --cfd를 붙이면 같은 설계를 CFD로 해석해 나란히 보여 준다 (OpenFOAM 필요)

사용법: uv run python ch10/define.py [--cfd]   (CFD 작업 폴더 ~/cfd-runs/ch10/range)
"""
import sys

import numpy as np

from problem import (BASE, BOUNDS, GAP_MIN, NAMES, base_values, cfd, gap,
                     grid, surrogate)

b = base_values()
print("기준 설계  " + "  ".join(f"{k} {v:g}" for k, v in zip(NAMES, BASE)))
print(f"  dp {b['dp']:.5f} m²/s²  dT_max {b['dT_max']:.3f} K"
      f"  dT_avg {b['dT_avg']:.3f} K  P {b['P']:.3f} mW/m")

x = grid()
s = surrogate(x)
rules = [("틈 H-e ≥ 1.0mm", gap(x) >= GAP_MIN),
         ("P ≤ 기준 설계", s["P"] <= b["P"]),
         ("미수렴 모서리 밖", ~((x[:, 1] > 0.355) & (x[:, 3] > 0.117)
                          & (x[:, 2] <= 9)))]
print(f"\n격자 {len(x):,}개 가운데 남는 비율")
print("   단독    누적   제약")
keep = np.ones(len(x), bool)
for name, m in rules:
    keep &= m
    print(f"  {m.mean() * 100:5.1f}%  {keep.mean() * 100:5.1f}%   {name}")

print("\n학습 범위 밖으로 나갈 때 (나머지 변수는 기준 설계)")
xs = []
for i, vals in [(1, [0.3, 0.4, 0.5, 0.6]), (2, [8, 15, 20, 30])]:
    for v in vals:
        xi = BASE.copy()
        xi[i] = v
        xs.append((i, v, xi))
X = np.array([xi for _, _, xi in xs])
pred = surrogate(X)
truth = cfd(X, "~/cfd-runs/ch10/range") if "--cfd" in sys.argv else None
print("  변수       값  범위   예측 dT_max      P"
      + ("   CFD dT_max      P" if truth else ""))
for k, (i, v, _) in enumerate(xs):
    inside = BOUNDS[i, 0] <= v <= BOUNDS[i, 1]
    line = (f"  {NAMES[i]:<8} {v:5g}   {'안' if inside else '밖'}"
            f"       {pred['dT_max'][k]:6.3f} {pred['P'][k]:6.2f}")
    if truth:
        r = truth[k]
        line += (f"       {r['dT_max']:6.3f} {r['P']:6.2f}" if r["dT_max"]
                 else f"   {r['status']}")
    print(line)
