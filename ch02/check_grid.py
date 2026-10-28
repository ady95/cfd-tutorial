"""책 02-5: grid_run.sh로 해석한 격자 네 개의 결과를 비교한다.

사용법: uv run python ch02/check_grid.py ~/cfd-runs/grid
결과:   표 출력과 작업 폴더의 grid-convergence.png
"""
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

U_MEAN = 0.1    # 입구 평균 속도 [m/s]
H = 0.002       # 채널 높이 [m]
NU = 1e-06      # 동점성계수 [m2/s]
U_MAX = 1.5 * U_MEAN
DPDX = 12 * NU * U_MEAN / H**2

root = Path(sys.argv[1]).expanduser()
cases = sorted(root.glob("ny*"), key=lambda c: int(c.name[2:]))


def read_case(case):
    """마지막 반복의 내부 격자와 입구 경계면을 읽는다."""
    foam = case / "case.foam"
    foam.touch()
    reader = pv.POpenFOAMReader(str(foam))
    reader.set_active_time_value(reader.time_values[-1])
    data = reader.read()
    return (reader.time_values[-1], data["internalMesh"],
            data["boundary"]["inlet"])


rows = []
for case in cases:
    it, mesh, inlet = read_case(case)
    x, y = mesh.cell_centers().points[:, :2].T
    ux, p = mesh.cell_data["U"][:, 0], mesh.cell_data["p"]
    xs, ys = np.unique(x.round(9)), np.unique(y.round(9))

    # 1. x = 95 mm에 가장 가까운 열: 셀 중심 높이에서 이론해와 비교
    col = np.isclose(x, xs[np.argmin(abs(xs - 0.095))])
    u_th = 6 * U_MEAN * (y[col] / H) * (1 - y[col] / H)
    prof = abs(ux[col] - u_th).max() / U_MAX * 100

    # 2. 채널 가운데에 가장 가까운 행: x = 50~95 mm 압력 기울기
    row = np.isclose(y, ys[np.argmin(abs(ys - H / 2))])
    zone = row & (x >= 0.05) & (x <= 0.095)
    dpdx = -np.polyfit(x[zone], p[zone], 1)[0]

    # 3. x = 10 mm 단면 평균 압력 (열마다 평균한 뒤 보간)
    p_col = [p[np.isclose(x, xc)].mean() for xc in xs]
    p10 = np.interp(0.01, xs, p_col)

    # 4. 입구 면 평균 압력 (patchAverage 기록의 마지막 줄)
    dat = next(case.glob("postProcessing/patchAverage*/0/*.dat"))
    p_in = np.loadtxt(dat)[-1, 1]

    # 5. 입구 면의 압력: 벽 옆 면과 가운데 면
    idx = np.argsort(inlet.cell_centers().points[:, 1])
    p_face = inlet.cell_data["p"][idx]
    p_wall, p_mid = p_face[0], p_face[len(idx) // 2]

    log = (case / "log.foamRun").read_text()
    sec = float(re.findall(r"ExecutionTime = ([\d.]+)", log)[-1])
    rows.append(dict(ny=int(case.name[2:]), cells=len(x), it=it, sec=sec,
                     prof=prof, slope=(dpdx / DPDX - 1) * 100,
                     p10=p10, p_in=p_in, p_wall=p_wall, p_mid=p_mid))

print("격자    셀 수  반복  시간[s]  분포 오차  기울기 오차"
      "  p(10mm)   입구 p")
for r in rows:
    print(f"ny{r['ny']:3d} {r['cells']:7d} {r['it']:5.0f} {r['sec']:8.2f}"
          f" {r['prof']:9.3f}% {r['slope']:+11.3f}%"
          f" {r['p10']:9.5f} {r['p_in']:8.5f}")

print("\n앞 격자 대비 변화율과 관찰 차수 (마지막 세 격자)")
for key, name in [("p10", "p(10mm) "), ("p_in", "입구 p  ")]:
    f = np.array([r[key] for r in rows])
    change = np.diff(f) / f[1:] * 100
    order = np.log((f[-2] - f[-3]) / (f[-1] - f[-2])) / np.log(2)
    print(name, "  ".join(f"{c:+.2f}%" for c in change),
          f"  관찰 차수 {order:.2f}")

print("\n입구 면 압력    벽 옆   가운데")
for r in rows:
    print(f"ny{r['ny']:3d}       {r['p_wall']:8.5f} {r['p_mid']:8.5f}")

# 그림: (a) 이론해 대비 오차, (b) 앞 격자 대비 변화율
dy = np.array([H / r["ny"] * 1000 for r in rows])
cells = np.array([r["cells"] for r in rows])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.loglog(dy, [r["prof"] for r in rows], "o-", color="#2a78d6",
           lw=2, ms=8, label="velocity profile")
ax1.loglog(dy, [abs(r["slope"]) for r in rows], "s-", color="#eb6834",
           lw=2, ms=8, label="pressure gradient")
ref = 0.5 * rows[0]["prof"] * (dy / dy[0]) ** 2   # 겹치지 않게 아래로
ax1.loglog(dy, ref, "--", color="0.6", lw=1.5, label="slope 2 (2nd order)")
ax1.set_xlabel("cell height [mm]")
ax1.set_ylabel("error vs theory [%]")
ax1.set_title("(a) fully developed region")

for key, mark, color, name in [("p10", "o-", "#2a78d6", "p at x = 10 mm"),
                               ("p_in", "s-", "#eb6834", "inlet average p")]:
    f = np.array([r[key] for r in rows])
    ax2.semilogx(cells[1:], np.diff(f) / f[1:] * 100, mark, color=color,
                 lw=2, ms=8, label=name)
ax2.axhline(1.0, ls="--", color="0.6", lw=1.5)
ax2.set_xlabel("number of cells (finer grid of each pair)")
ax2.set_ylabel("change from previous grid [%]")
ax2.set_title("(b) pressure drop")
ax2.set_ylim(bottom=0)

ax1.set_xticks(dy, [f"{d:g}" for d in dy])
ax2.set_xticks(cells[1:], [f"{c:,}" for c in cells[1:]])
for ax in (ax1, ax2):
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(root / "grid-convergence.png", dpi=150)
print("\n저장:", root / "grid-convergence.png")
