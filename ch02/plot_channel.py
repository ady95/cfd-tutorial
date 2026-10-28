"""책 02-4: channel2d 결과를 PyVista로 읽어 그림 두 장을 만든다.

사용법: uv run python ch02/plot_channel.py ~/cfd-runs/channel2d
결과:   케이스 폴더의 images/ 아래 channel2d-ux.png, channel2d-profiles.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

U_MEAN = 0.1    # 입구 평균 속도 [m/s]
H = 0.002       # 채널 높이 [m]
Y_SCALE = 10    # 그림에서 높이 방향 확대 배율

case = Path(sys.argv[1]).expanduser()
foam = case / "case.foam"
foam.touch()    # OpenFOAM 리더가 케이스를 알아보는 빈 파일
out = case / "images"
out.mkdir(exist_ok=True)

reader = pv.POpenFOAMReader(str(foam))
last = reader.time_values[-1]
reader.set_active_time_value(last)
mesh = reader.read()["internalMesh"]
print(f"반복 {last:g}회 결과, 셀 {mesh.n_cells}개")
print("셀 데이터:", list(mesh.cell_data.keys()))

# 1. x 방향 속도장: 채널이 50:1로 가늘어 높이 방향을 늘려 그린다
view = mesh.scale([1, Y_SCALE, 1])
view.point_data["Ux"] = view.point_data["U"][:, 0]
pl = pv.Plotter(off_screen=True, window_size=(1600, 500))
pl.add_mesh(
    view, scalars="Ux", cmap="viridis", lighting=False,
    scalar_bar_args={"title": "Ux [m/s]", "n_labels": 4, "fmt": "%.3f",
                     "position_x": 0.25, "position_y": 0.03,
                     "width": 0.5, "height": 0.1},
)
pl.view_xy()
pl.camera.zoom(3.0)     # 화면 폭에 맞게 확대
pl.screenshot(out / "channel2d-ux.png")

# 2. 단면별 속도 분포: 입구 구간에서 포물선으로 자라는 모습
fig, ax = plt.subplots(figsize=(7, 4.5))
for x_mm in [1, 5, 10, 20, 95]:
    x = x_mm / 1000
    line = mesh.sample_over_line((x, 0, 5e-5), (x, H, 5e-5), resolution=100)
    y = line.points[:, 1]
    ux = line.point_data["U"][:, 0]
    ax.plot(ux, y * 1000, label=f"x = {x_mm} mm")
    print(f"x = {x_mm:3d} mm  중심 속도 {ux[len(ux) // 2]:.5f} m/s")

y = np.linspace(0, H, 101)
ax.plot(6 * U_MEAN * (y / H) * (1 - y / H), y * 1000, "k--",
        label="theory (fully developed)")
ax.set_xlabel("Ux [m/s]")
ax.set_ylabel("y [mm]")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out / "channel2d-profiles.png", dpi=150)
print("저장:", sorted(p.name for p in out.glob("*.png")))
