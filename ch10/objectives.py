"""책 10-2: 냉각 성능과 펌핑 손실 지표를 정하고, 가중합 목적함수를 시험한다.

1) 압력손실 dp와 펌핑 동력 P가 설계 순위를 다르게 매기는 예
2) 가중합 J = dT_max/기준값 + w x P/기준값 을 w마다 격자에서 최소화
   (틈 제약만 적용) -> w에 따라 최적 설계가 어떻게 바뀌는지

사용법: uv run python ch10/objectives.py
"""
import numpy as np

from problem import GAP_MIN, base_values, gap, grid, outlet_rise, surrogate

b = base_values()
print("H 바꾸기 (e_ratio 0.3, N 8, U 0.1)")
print("    H    dp [m²/s²]  유량 [cm²/s]  P [mW/m]  dT_max  출구 상승")
for H in (1.5, 2.0, 2.5):
    x = np.array([H, 0.3, 8, 0.1])
    s = surrogate(x)
    print(f"  {H:3.1f}     {s['dp'][0]:.5f}       {H * 0.1 * 10:5.2f}"
          f"     {s['P'][0]:6.2f}   {s['dT_max'][0]:5.3f}"
          f"    {outlet_rise(H, 0.1):5.3f}")

x = grid()
s = surrogate(x)
ok = gap(x) >= GAP_MIN
print("\n가중합 최소화 (기준 설계 = J 1 + w)")
print("     w      H  e_ratio   N      U   dT_max      P      J")
for w in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
    J = s["dT_max"] / b["dT_max"] + w * s["P"] / b["P"]
    i = np.argmin(np.where(ok, J, np.inf))
    H, er, N, U = x[i]
    print(f"  {w:4.2f}   {H:4.2f}   {er:6.4f}  {N:2.0f}  {U:5.3f}"
          f"   {s['dT_max'][i]:6.3f}  {s['P'][i]:5.2f}  {J[i]:5.3f}")
