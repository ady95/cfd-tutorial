"""책 부록 A-4: 3장의 2D 리브 채널을 폭이 있는 3D 채널로 넓혀 해석한다.

채널 폭 WZ의 절반만 푼다. z=0은 대칭면(채널 가운데), z=WZ/2는 옆벽이다.
리브는 폭 전체를 가로지른다. 나머지 조건(유속, 가열, 물성, 함수 객체)은
ch03/template과 같다. 옆벽은 미끄럼 없음, 단열이다.

사용법: uv run python appendix/a4_3d_cht/channel3d.py [작업 폴더] [코어 수] [NZ]
    기본값 ~/cfd-runs/a4/3d  6  20
결과: 셀 수, 반복 수, 시간, dp·dT_max·dT_avg와 2D 값(03-2) 비교,
      가열면 온도의 폭 방향 분포
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ch03"))
from channel_mesh import DEPTH, block_mesh_dict  # noqa: E402
from extract import extract  # noqa: E402

H, E, N, U = 2.0, 0.6, 8, 0.1     # 03-2의 기준 형상
WZ = 10.0                         # 채널 폭 [mm] (높이의 5배)
NZ = 20                           # 폭 절반의 셀 수
GZ = 0.25                         # 옆벽 쪽 셀 크기 / 가운데 셀 크기
T_IN = 293.15
GRAD_T = 16666.67            # 가열면 온도 기울기 q/k [K/m] (03-2)
SYM = """    symmetry
    {
        type            symmetryPlane;
    }"""


def mesh_3d(H, e, N, nz):
    text, cells = block_mesh_dict(H, e, N)
    text = re.sub(rf"(\(\S+ \S+ ){DEPTH:g}\)", rf"\g<1>{WZ / 2:g})", text)
    hexes = re.findall(r"hex \(([\d ]+)\)", text)
    text = text.replace(" 1) simpleGrading (1 1 1)",
                        f" {nz}) simpleGrading (1 1 {GZ})")
    front, back = [], []
    for h in hexes:
        c = h.split()
        front.append(f"            ({c[0]} {c[3]} {c[2]} {c[1]})")
        back.append(f"            ({c[4]} {c[5]} {c[6]} {c[7]})")
    text = re.sub(r"defaultPatch\n\{[^}]*\}\n", "", text)
    extra = ("    symmetry\n    {\n        type symmetryPlane;\n"
             "        faces\n        (\n" + "\n".join(front) +
             "\n        );\n    }\n    sideWall\n    {\n        type wall;\n"
             "        faces\n        (\n" + "\n".join(back) +
             "\n        );\n    }\n);")
    text = text.rstrip().rstrip(");").rstrip() + "\n" + extra + "\n"
    return text, cells * nz


def make_case(case, nproc, nz):
    subprocess.run(["rm", "-rf", str(case)], check=True)
    subprocess.run(["cp", "-r", str(ROOT / "ch03/template"), str(case)],
                   check=True)
    text, cells = mesh_3d(H, E, N, nz)
    (case / "system/blockMeshDict").write_text(text)
    for f in ("U", "p", "T"):
        p = case / "0" / f
        p.write_text(re.sub(r"    frontAndBack\n    \{[^}]*\}", SYM,
                            p.read_text()))
    (case / "system/decomposeParDict").write_text(
        "FoamFile\n{\n    format ascii;\n    class dictionary;\n"
        "    object decomposeParDict;\n}\n"
        f"numberOfSubdomains {nproc};\ndecomposer scotch;\n")
    return cells


def sh(case, *cmd):
    name = cmd[0] if cmd[0] != "mpirun" else cmd[3]
    with open(case / f"log.{name}", "w") as f:
        subprocess.run(list(cmd), cwd=case, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


def wall_profile(case):
    """가열면 온도 상승을 폭 방향 위치(z)별 최댓값으로 정리한다."""
    import pyvista as pv
    foam = case / "case.foam"
    foam.touch()
    r = pv.POpenFOAMReader(str(foam))
    r.set_active_time_value(r.time_values[-1])
    r.enable_all_patch_arrays()
    data = r.read()
    mesh, wall = data["internalMesh"], data["boundary"]["heatedWall"]
    c = wall.cell_centers().points
    z = c[:, 2] * 1e3
    # 파일의 가열면 T는 옆 셀 값이다 (07-3). 셀 중심까지 거리만큼 올린다
    cell = mesh.find_containing_cell(c + [0, 1e-7, 0])
    y_cell = mesh.cell_centers().points[cell, 1]
    t = wall.cell_data["T"] - T_IN + GRAD_T * y_cell
    zs = np.unique(np.round(z, 4))
    return zs, np.array([t[np.isclose(z, v, atol=1e-4)].max() for v in zs])


if __name__ == "__main__":
    case = Path(sys.argv[1] if len(sys.argv) > 1
                else "~/cfd-runs/a4/3d").expanduser()
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    nz = int(sys.argv[3]) if len(sys.argv) > 3 else NZ
    case.parent.mkdir(parents=True, exist_ok=True)
    cells = make_case(case, nproc, nz)
    print(f"3D 채널: 폭 {WZ:g} mm의 절반, 셀 {cells:,}개 "
          f"(2D 셀 수 x {nz}), 코어 {nproc}개")
    start = time.perf_counter()
    sh(case, "blockMesh")
    if nproc > 1:
        sh(case, "decomposePar")
        sh(case, "mpirun", "-np", str(nproc), "foamRun", "-parallel")
        sh(case, "reconstructPar", "-latestTime")
    else:
        sh(case, "foamRun")
    sec = time.perf_counter() - start
    r = extract(case)
    print(f"반복 {r['iterations']} ({'수렴' if r['converged'] else '미수렴'})"
          f", 걸린 시간 {sec:.0f}초")
    book = {"dp": 0.04228, "dT_max": 7.6666, "dT_avg": 5.356}
    for k, v in book.items():
        print(f"{k:<7} 3D {r[k]:.5g}   2D {v:.5g}   "
              f"차이 {(r[k] / v - 1) * 100:+.1f}%")
    zs, tmax = wall_profile(case)
    print("\n가열면 최고 온도 상승의 폭 방향 분포 (z=0 채널 가운데, "
          f"z={WZ / 2:g} 옆벽)")
    for i in np.linspace(0, len(zs) - 1, 11).astype(int):
        print(f"  z {zs[i]:5.2f} mm  {tmax[i]:.3f} K")
    i = int(np.argmax(tmax))
    print(f"가장 뜨거운 줄: z {zs[i]:.2f} mm, {tmax[i]:.3f} K")
    np.savetxt(case / "wall_profile.csv", np.column_stack([zs, tmax]),
               delimiter=",", header="z_mm,dT_max_K", fmt="%.4f")
