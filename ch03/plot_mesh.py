"""책 03-1: blockMesh로 만든 리브 채널 격자의 일부를 그림으로 저장한다.

사용법: uv run python ch03/plot_mesh.py ~/cfd-runs/rib-check/H2_e1_N8
결과:   케이스 폴더의 images/mesh.png (첫 번째 리브 부근)
"""
import sys
from pathlib import Path

import pyvista as pv

X0, X1 = 11.0, 19.0      # 그릴 구간 [mm]: 첫 번째 리브 부근
COLORS = {"heatedWall": "#eb6834", "ribWall": "#2a78d6",
          "upperWall": "black", "lowerWall": "black"}

case = Path(sys.argv[1]).expanduser()
foam = case / "case.foam"
foam.touch()
out = case / "images"
out.mkdir(exist_ok=True)

reader = pv.POpenFOAMReader(str(foam))
reader.enable_all_patch_arrays()
reader.disable_all_cell_arrays()   # 격자만 그리므로 필드 값은 읽지 않는다
data = reader.read()
box = (X0 / 1000, X1 / 1000, -1, 1, -1, 1)

pl = pv.Plotter(off_screen=True, window_size=(1600, 460))
inner = data["internalMesh"].clip_box(box, invert=False)
pl.add_mesh(inner, color="white", show_edges=True, edge_color="#8c8c8c",
            line_width=1, lighting=False)
for name, color in COLORS.items():
    patch = data["boundary"][name]
    if patch.n_cells == 0:
        continue
    part = patch.clip_box(box, invert=False)
    if part.n_cells:
        pl.add_mesh(part.extract_feature_edges(boundary_edges=True),
                    color=color, line_width=5)
pl.view_xy()
pl.camera.tight(padding=0.03)   # 그린 구간이 화면을 채우게
pl.set_background("white")
pl.screenshot(out / "mesh.png")
print("저장:", out / "mesh.png")
