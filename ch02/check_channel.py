"""책 02-3: 2D 채널 해석 결과를 평행 평판 층류 이론해와 비교한다.

사용법: uv run python ch02/check_channel.py ~/cfd-runs/channel2d
"""
import sys
from pathlib import Path

import numpy as np

U_MEAN = 0.1    # 입구 평균 속도 [m/s]
H = 0.002       # 채널 높이 [m]
L = 0.1         # 채널 길이 [m]
NU = 1e-06      # 동점성계수 [m2/s]

case = Path(sys.argv[1]).expanduser()
post = case / "postProcessing"


def latest(name):
    """postProcessing/<name>/ 아래 마지막 반복 폴더의 line.xy를 읽는다."""
    dirs = sorted((post / name).iterdir(), key=lambda d: float(d.name))
    return dirs[-1].name, np.loadtxt(dirs[-1] / "line.xy")


it, prof = latest("outletProfile")
_, line = latest("centerline")
print(f"반복 {it}회 결과, Re(높이 기준) = {U_MEAN * H / NU:.0f}")

# 1. 출구 근처(x = 95 mm) 속도 분포: 이론해 u = 6 U (y/H) (1 - y/H)
y, ux = prof[:, 0], prof[:, 1]
u_theory = 6 * U_MEAN * (y / H) * (1 - y / H)
u_max = ux.max()
err = (u_max - 1.5 * U_MEAN) / (1.5 * U_MEAN) * 100
print(f"최대 속도      CFD {u_max:.5f}  이론 {1.5 * U_MEAN:.5f} m/s"
      f"  오차 {err:+.2f}%")
dev = np.abs(ux - u_theory).max() / (1.5 * U_MEAN) * 100
print(f"속도 분포      이론해와 최대 차이 {dev:.2f}% (최대 속도 대비)")

# 2. 발달 구간(x = 50~95 mm)의 압력 기울기: 이론 12 nu U / H^2
x, uc, p = line[:, 0], line[:, 1], line[:, 4]
dev_zone = (x >= 0.05) & (x <= 0.095)
dpdx = -np.polyfit(x[dev_zone], p[dev_zone], 1)[0]
dpdx_theory = 12 * NU * U_MEAN / H**2
err = (dpdx - dpdx_theory) / dpdx_theory * 100
print(f"압력 기울기    CFD {dpdx:.5f}  이론 {dpdx_theory:.5f} m/s2"
      f"  오차 {err:+.2f}%")

# 3. 입구 구간 길이: 중심 속도가 출구 쪽 값의 99%에 처음 닿는 x
x99 = x[np.argmax(uc >= 0.99 * uc[-1])]
print(f"입구 구간 길이 {x99 * 1000:.1f} mm (중심 속도 99% 도달)")

# 4. 입출구 압력차(입구 평균 압력, 출구 p = 0): 완전발달 이론값과 비교
avg = post / "patchAverage(patch=inlet,fields=(p))" / "0"
dp_inlet = np.loadtxt(avg / "surfaceFieldValue.dat")[-1, 1]
print(f"입출구 압력차  CFD {dp_inlet:.5f}  완전발달 가정 "
      f"{dpdx_theory * L:.5f} m2/s2")
