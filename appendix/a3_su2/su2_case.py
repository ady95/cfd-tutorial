"""책 부록 A-3: 3장의 냉각 채널 케이스를 SU2로 풀어 OpenFOAM과 비교한다.

1) ch03의 템플릿과 channel_mesh.py로 OpenFOAM 격자를 만든다 (blockMesh만)
2) 그 격자를 SU2 2D 격자(.su2)로 바꾼다. 앞면(z=0)의 사각형 면이 SU2 요소가 된다
3) 3장과 같은 조건(물 물성 일정, 층류, 입구 0.1 m/s·20 °C, 가열면 10 kW/m²)으로
   SU2의 비압축성 솔버(INC_NAVIER_STOKES, 에너지 방정식 포함)를 돌린다
4) 가열면 최고·평균 온도 상승과 입구 평균 압력을 뽑는다

SU2는 격자점(노드)에 값을 두는 방식이라 가열면 노드의 온도를 그대로 읽는다.
OpenFOAM은 셀 중심에 값을 두므로 같은 격자라도 미지수의 위치가 다르다.

사용법: uv run python appendix/a3_su2/su2_case.py H e N U [작업 폴더]
환경 변수 SU2_HOME에 SU2를 푼 폴더를 준다 (기본 ~/opt/su2-8.5.0).
"""
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ch03"))
import channel_mesh  # noqa: E402

SU2 = Path(os.environ.get("SU2_HOME", "~/opt/su2-8.5.0")).expanduser()
RHO, CP, MU = 998.0, 4182.0, 9.98e-4
K = 1.4376e-7 * RHO * CP           # 3장의 열확산계수와 같게: k = alpha rho cp
Q, T_IN = 1e4, 293.15
CFL_DEFAULT = 1000          # 100보다 반복이 7분의 1, 결과는 7자리까지 같음
PATCHES = ["inlet", "outlet", "lowerWall", "heatedWall", "upperWall",
           "ribWall"]

CONFIG = """\
SOLVER= INC_NAVIER_STOKES
KIND_TURB_MODEL= NONE
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

INC_NONDIM= DIMENSIONAL
FLUID_MODEL= CONSTANT_DENSITY
INC_DENSITY_INIT= {rho}
INC_VELOCITY_INIT= ( {u}, 0.0, 0.0 )
INC_ENERGY_EQUATION= YES
INC_TEMPERATURE_INIT= {t_in}
SPECIFIC_HEAT_CP= {cp}
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= {mu}
CONDUCTIVITY_MODEL= CONSTANT_CONDUCTIVITY
THERMAL_CONDUCTIVITY_CONSTANT= {k}

INC_INLET_TYPE= VELOCITY_INLET
MARKER_INLET= ( inlet, {t_in}, {u}, 1.0, 0.0, 0.0 )
INC_OUTLET_TYPE= PRESSURE_OUTLET
MARKER_OUTLET= ( outlet, 0.0 )
MARKER_HEATFLUX= ( heatedWall, {q}, lowerWall, 0.0, upperWall, 0.0, ribWall, 0.0 )
MARKER_PLOTTING= ( heatedWall )
MARKER_MONITORING= ( heatedWall )
MARKER_ANALYZE= ( inlet, outlet )
MARKER_ANALYZE_AVERAGE= AREA

NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= {cfl}
CFL_ADAPT= NO
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= NONE
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-8
LINEAR_SOLVER_ITER= 20

ITER= {iters}
CONV_FIELD= RMS_PRESSURE
CONV_RESIDUAL_MINVAL= -11
CONV_STARTITER= 10

MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
OUTPUT_FILES= ( RESTART_ASCII, SURFACE_CSV )
SURFACE_FILENAME= surface
SCREEN_OUTPUT= ( INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X, RMS_TEMPERATURE )
HISTORY_OUTPUT= ( ITER, RMS_RES, FLOW_COEFF )
SCREEN_WRT_FREQ_INNER= 100
"""


def read_list(path):
    """OpenFOAM ASCII 목록 파일의 본문(괄호 안)을 줄 단위로."""
    text = path.read_text()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    m = re.search(r"\n(\d+)\s*\n\(\n", text)
    body = text[m.end():].rsplit(")", 1)[0]
    return body.strip().splitlines()


def boundary(poly):
    text = (poly / "boundary").read_text()
    out = {}
    for name, body in re.findall(r"\n\s*(\w+)\s*\n\s*\{([^}]*)\}", text):
        n = re.search(r"nFaces\s+(\d+)", body)
        s = re.search(r"startFace\s+(\d+)", body)
        if n and s:
            out[name] = (int(s.group(1)), int(n.group(1)))
    return out


def to_su2(case, out):
    """OpenFOAM 2D 격자(앞뒤 empty 한 층) -> SU2 2D 격자. 노드 수를 돌려준다."""
    poly = case / "constant/polyMesh"
    pts = np.array([[float(v) for v in ln.strip("()").split()]
                    for ln in read_list(poly / "points")])
    faces = [[int(v) for v in ln[ln.index("(") + 1:-1].split()]
             for ln in read_list(poly / "faces")]
    bnd = boundary(poly)
    front = np.isclose(pts[:, 2], 0.0)
    new = -np.ones(len(pts), int)
    new[front] = np.arange(front.sum())

    s, n = bnd["frontAndBack"]
    elems = []
    for f in faces[s:s + n]:
        if not all(front[f]):
            continue
        xy = pts[f, :2]
        area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                            - np.roll(xy[:, 0], -1) * xy[:, 1])
        elems.append([new[i] for i in (f if area > 0 else f[::-1])])

    lines = [f"NDIME= 2", f"NELEM= {len(elems)}"]
    lines += [f"9 {a} {b} {c} {d} {i}" for i, (a, b, c, d) in
              enumerate(elems)]
    lines += [f"NPOIN= {front.sum()}"]
    lines += [f"{x:.9e} {y:.9e} {i}" for i, (x, y) in
              enumerate(pts[front, :2])]
    lines += [f"NMARK= {len(PATCHES)}"]
    for name in PATCHES:
        s, n = bnd[name]
        segs = [[new[i] for i in f if front[i]] for f in faces[s:s + n]]
        lines += [f"MARKER_TAG= {name}", f"MARKER_ELEMS= {len(segs)}"]
        lines += [f"3 {a} {b}" for a, b in segs]
    out.write_text("\n".join(lines) + "\n")
    return int(front.sum()), len(elems)


def enthalpy(csv_path):
    rows = list(csv.reader(open(csv_path)))
    head = [h.strip().strip('"') for h in rows[0]]
    ix, iy, ih = (head.index(k) for k in ("x", "y", "Enthalpy"))
    return np.array([[float(r[ix]), float(r[iy]), float(r[ih])]
                     for r in rows[1:]])


def along(su2):
    """가열면 노드의 (x, 온도 상승). SU2는 엔탈피를 풀므로 cp가 일정하면
    온도 상승 = (h - 입구 h) / cp 이다. 입구 h는 restart.csv에서 읽는다."""
    vol = enthalpy(su2 / "restart.csv")
    h_in = vol[np.isclose(vol[:, 0], 0.0), 2].mean()
    wall = enthalpy(su2 / "surface.csv")
    k = np.argsort(wall[:, 0])
    return wall[k, 0], (wall[k, 2] - h_in) / CP


def history_last(case):
    rows = list(csv.reader(open(case / "history.csv")))
    head = [h.strip().strip('"') for h in rows[0]]
    return dict(zip(head, (float(v) for v in rows[-1])))


def run(H, e, N, U, case, iters=5000, cells_h=30, cfl=CFL_DEFAULT,
        threads=1):
    case = Path(case)
    subprocess.run(["rm", "-rf", str(case)], check=True)
    subprocess.run(["cp", "-r", str(ROOT / "ch03/template"), str(case)],
                   check=True)
    channel_mesh.CELLS_H = cells_h       # 높이 방향 셀 수 (03-5의 격자 검사)
    text, _ = channel_mesh.block_mesh_dict(H, e, N)
    (case / "system/blockMeshDict").write_text(text)
    with open(case / "log.blockMesh", "w") as f:
        subprocess.run(["blockMesh"], cwd=case, stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    su2 = case / "su2"
    su2.mkdir()
    nodes, elems = to_su2(case, su2 / "mesh.su2")
    (su2 / "case.cfg").write_text(CONFIG.format(
        rho=RHO, u=U, t_in=T_IN, cp=CP, mu=MU, k=f"{K:.6g}", q=Q,
        iters=iters, cfl=cfl))
    start = time.perf_counter()
    with open(su2 / "log.SU2_CFD", "w") as f:
        subprocess.run([str(SU2 / "bin/SU2_CFD"), "case.cfg"], cwd=su2,
                       stdout=f, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "OMP_NUM_THREADS": str(threads)})
    sec = time.perf_counter() - start
    x, t = along(su2)
    hist = history_last(su2)
    # 가열면 평균은 노드 사이 길이로 가중한 사다리꼴 평균
    t_avg = np.trapezoid(t, x) / (x[-1] - x[0])
    # MARKER_ANALYZE의 Pressure_Drop = 출구 평균 - 입구 평균 (출구 압력 0)
    p_in = -hist["Pressure_Drop"]
    return {"nodes": nodes, "elems": elems, "seconds": sec,
            "iters": int(hist["Inner_Iter"]) + 1,
            "rms_p": hist["rms[P]"], "dT_max": t.max(),
            "dT_avg": t_avg, "p_inlet": p_in / RHO}


if __name__ == "__main__":
    H, e, N, U = (float(v) for v in sys.argv[1:5])
    case = Path(sys.argv[5] if len(sys.argv) > 5
                else "~/cfd-runs/a3/su2-base").expanduser()
    r = run(H, e, int(N), U, case)
    print(f"SU2 격자: 노드 {r['nodes']:,}개, 요소 {r['elems']:,}개")
    print(f"반복 {r['iters']} (압력 잔차 1e{r['rms_p']:.1f}), "
          f"걸린 시간 {r['seconds']:.1f}초 (스레드 하나)")
    print(f"입구 평균 압력 {r['p_inlet']:.5f} m2/s2")
    print(f"가열면 최고 온도 상승 {r['dT_max']:.4f} K, "
          f"평균 {r['dT_avg']:.4f} K")
