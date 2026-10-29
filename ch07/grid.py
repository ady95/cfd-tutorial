"""책 07-3: OpenFOAM 해석 결과를 FNO가 읽는 균일 격자 배열로 옮긴다.

케이스마다 격자 모양이 다르므로(H, e, N에 따라 셀 배치가 바뀜) 모든 케이스를
같은 크기의 균일 격자에 보간한다. 격자는 x 0~80 mm, y 0~3 mm를 덮는다.
범위 밖 데이터(ood_H)의 H가 최대 3 mm이기 때문이다.

- 보간 재료: 셀 중심 값 + 벽·입구·출구 면 중심 값
- 가열면 온도는 파일에 저장되지 않으므로(fixedGradient) 옆 셀 값에
  기울기 x 거리를 더해 직접 계산한다
- 유체 밖(리브 속, 채널 위 빈 공간)은 마스크 0, 값 0

다른 스크립트에서는 to_grid(케이스 폴더, H, e, N, nx, ny)를 쓴다.
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch03"))
from channel_mesh import L_HEAT, L_IN, rib_spans  # noqa: E402

LX, LY = 80.0, 3.0          # 격자가 덮는 영역 [mm]
T_IN = 293.15               # 입구 온도 [K]
GRAD_T = 16666.67           # 가열면 온도 기울기 q/k [K/m] (03-2)
FIELDS = ["Ux", "Uy", "p", "dT"]
WALLS = ["inlet", "outlet", "lowerWall", "heatedWall", "upperWall", "ribWall"]


def axes(nx, ny):
    """격자점 좌표 [mm]. 양 끝(입구·출구, 아래 벽)을 격자점에 포함한다."""
    return np.linspace(0, LX, nx), np.linspace(0, LY, ny)


def fluid_mask(H, e, N, nx, ny):
    """격자점이 유체 안(벽 위 포함)이면 1, 리브 속이나 채널 밖이면 0."""
    x, y = axes(nx, ny)
    X, Y = np.meshgrid(x, y)                     # 모양 (ny, nx)
    tol = 1e-9
    fluid = Y <= H + tol
    for a, b in rib_spans(N):
        rib = (X > a + tol) & (X < b - tol) & (Y > H - e + tol)
        fluid &= ~rib
    return fluid.astype(np.float32)


def read_points(case):
    """셀 중심과 경계면 중심의 (x, y) [mm]와 값 (Ux, Uy, p, dT)."""
    foam = Path(case) / "case.foam"
    foam.touch()
    reader = pv.POpenFOAMReader(str(foam))
    reader.enable_all_patch_arrays()
    reader.set_active_time_value(reader.time_values[-1])
    data = reader.read()
    mesh = data["internalMesh"]

    def values(block):
        u, p, t = (block.cell_data[k] for k in ("U", "p", "T"))
        return np.column_stack([u[:, 0], u[:, 1], p, t - T_IN])

    pts = [mesh.cell_centers().points]
    vals = [values(mesh)]
    for name in WALLS:
        patch = data["boundary"][name]
        c = patch.cell_centers().points
        v = values(patch)
        if name == "heatedWall":
            # 파일의 가열면 T는 옆 셀 값이다. 셀 중심에서 벽까지 거리만큼 올린다
            cell = mesh.find_containing_cell(c + [0, 1e-7, 0])
            y_cell = mesh.cell_centers().points[cell, 1]
            v[:, 3] = v[:, 3] + GRAD_T * y_cell
        pts.append(c)
        vals.append(v)
    xy = np.concatenate(pts)[:, :2] * 1000       # m -> mm
    return xy, np.concatenate(vals)


def to_grid(case, H, e, N, nx, ny):
    """케이스 하나를 (마스크 (ny, nx), 물리장 (4, ny, nx)) 배열로."""
    xy, v = read_points(case)
    x, y = axes(nx, ny)
    X, Y = np.meshgrid(x, y)
    q = np.column_stack([X.ravel(), Y.ravel()])
    out = LinearNDInterpolator(xy, v)(q)
    miss = np.isnan(out[:, 0])                   # 점 구름 바깥(모서리 몇 곳)
    out[miss] = NearestNDInterpolator(xy, v)(q[miss])
    mask = fluid_mask(H, e, N, nx, ny)
    fields = out.T.reshape(len(FIELDS), ny, nx) * mask
    return mask, fields.astype(np.float32)


def scalars(mask, fields):
    """격자 배열에서 3장의 세 값을 다시 계산한다 (dp, dT_max, dT_avg).

    dp: x = 5 mm 단면 유체 격자점의 평균 압력 (출구 압력 0)
    dT_max, dT_avg: 가열면(y = 0, x 10~70 mm) 격자점의 최고·평균
    """
    ny, nx = mask.shape
    x, _ = axes(nx, ny)
    i5 = np.argmin(np.abs(x - 5.0))
    col = mask[:, i5] > 0
    dp = fields[2, col, i5].mean()
    heated = (x >= L_IN - 1e-9) & (x <= L_IN + L_HEAT + 1e-9)
    wall = fields[3, 0, heated]
    return dp, wall.max(), wall.mean()
