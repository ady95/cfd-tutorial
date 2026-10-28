"""책 03-2: 리브 채널 해석 결과의 속도장과 온도장을 그린다.

사용법: uv run python ch03/plot_fields.py ~/cfd-runs/rib-base
결과:   케이스 폴더의 images/fields.png (가열 구간 앞쪽 x = 8~40 mm)
"""
import sys
from pathlib import Path

import pyvista as pv

X0, X1 = 8.0, 40.0      # 그릴 구간 [mm]
Y_SCALE = 4             # 높이 방향 확대 배율

case = Path(sys.argv[1]).expanduser()
foam = case / "case.foam"
foam.touch()
out = case / "images"
out.mkdir(exist_ok=True)

reader = pv.POpenFOAMReader(str(foam))
reader.set_active_time_value(reader.time_values[-1])
mesh = reader.read()["internalMesh"]
mesh = mesh.clip_box((X0 / 1000, X1 / 1000, -1, 1, -1, 1), invert=False)
mesh = mesh.cell_data_to_point_data().scale([1, Y_SCALE, 1])
mesh.point_data["|U|"] = (mesh.point_data["U"] ** 2).sum(axis=1) ** 0.5
mesh.point_data["T [C]"] = mesh.point_data["T"] - 273.15
print(f"반복 {reader.time_values[-1]:g}회 결과, "
      f"최고 온도 {mesh.point_data['T [C]'].max():.2f} C (그린 구간)")

pl = pv.Plotter(off_screen=True, shape=(2, 1), window_size=(1600, 1100))
panels = [("|U|", "viridis", "U magnitude [m/s]", "%.2f"),
          ("T [C]", "inferno", "T [C]", "%.1f")]
for row, (name, cmap, title, fmt) in enumerate(panels):
    pl.subplot(row, 0)
    lo = 20.0 if name == "T [C]" else 0.0
    pl.add_mesh(mesh, scalars=name, cmap=cmap, lighting=False,
                clim=(lo, mesh.point_data[name].max()),
                scalar_bar_args={"title": title, "n_labels": 4,
                                 "fmt": fmt, "position_x": 0.3,
                                 "position_y": 0.02, "width": 0.4,
                                 "height": 0.08, "color": "black"})
    pl.view_xy()
    pl.camera.tight(padding=0.02, adjust_render_window=False)
    pl.camera.zoom(0.9)     # 아래쪽에 범례 자리를 남긴다
pl.set_background("white")
pl.screenshot(out / "fields.png")
print("저장:", out / "fields.png")
