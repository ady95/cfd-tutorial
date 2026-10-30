"""책 부록 A-6: 구멍 뚫린 평판의 인장 해석 케이스를 만들고 돌리고 값을 뽑는다.

OpenFOAM 튜토리얼 solidDisplacement/plateHole을 매개변수화했다.
대칭을 이용해 판의 1/4만 푼다 (왼쪽 x=0, 아래 y=0이 대칭면).
- 판 반폭 W = 2 m 고정 (전체 폭 4 m), 오른쪽 끝에 인장 응력 SIGMA0
- 설계 변수: 구멍 반지름 비 r = R/W, 판 높이 비 h = Hp/W, 푸아송비 nu
- 결과: 응력 집중 계수 Kt = 구멍 위쪽 끝 sigma_xx / SIGMA0
        유연도 C = 오른쪽 끝 평균 변위 x E / (SIGMA0 x W)

사용법: python3 appendix/a6_structure/plate_case.py 0.25 1.0 0.3 [폴더] [ref]
다른 스크립트에서는 run_plate(폴더, r, h, nu, ref)를 쓴다.
"""
import math
import re
import subprocess
import sys
import time
from pathlib import Path

W = 2.0                     # 판 반폭 [m]
SIGMA0 = 1e4                # 오른쪽 끝 인장 응력 [Pa]
E = 2e11                    # 탄성계수 [Pa] (강철)
GRADE = 4                   # 구멍 둘레 반지름 방향 셀 크기 비 (바깥/안쪽)
ITER = 1000                # 의사 시간 반복 수 (절반 시점 값과 비교해 수렴 확인)
TEMPLATE = "/opt/openfoam14/tutorials/solidDisplacement/plateHole"


def block_mesh(R, Hp, ref):
    """튜토리얼과 같은 5블록 구조.

    구멍 둘레(R -> 2R)는 구멍 크기와 상관없이 같은 셀 수로 나누고, 반지름
    방향 셀을 구멍 쪽에 몰아 응력 집중을 잡는다 (바깥 셀이 안쪽의 GRADE배).
    바깥 블록은 셀 크기를 일정하게 둔다.
    """
    dx = 0.05 / ref
    c = math.sqrt(0.5)
    n_arc = 10 * ref                              # 45도 호 하나의 셀 수
    n_rad = 10 * ref                              # 구멍 R -> 2R
    n_x = max(4, math.ceil((W - 2 * R) / dx))     # 2R -> W
    n_y = max(4, math.ceil((Hp - 2 * R * c) / dx))
    g = GRADE
    v = [(R, 0), (2 * R, 0), (W, 0), (W, 2 * R * c), (2 * R * c, 2 * R * c),
         (R * c, R * c), (W, Hp), (2 * R * c, Hp), (0, Hp), (0, 2 * R),
         (0, R)]
    verts = "\n".join(f"    ({x:.6f} {y:.6f} 0)" for x, y in v)
    verts += "\n" + "\n".join(f"    ({x:.6f} {y:.6f} 0.5)" for x, y in v)
    a1, a2 = math.cos(math.pi / 8), math.sin(math.pi / 8)

    def arc(i, j, r, z, swap=False):
        x, y = (a2, a1) if swap else (a1, a2)
        return f"    arc {i} {j} ({r * x:.6f} {r * y:.6f} {z})"

    edges = "\n".join([arc(0, 5, R, 0), arc(5, 10, R, 0, True),
                       arc(1, 4, 2 * R, 0), arc(4, 9, 2 * R, 0, True),
                       arc(11, 16, R, 0.5), arc(16, 21, R, 0.5, True),
                       arc(12, 15, 2 * R, 0.5),
                       arc(15, 20, 2 * R, 0.5, True)])
    blocks = f"""    hex (5 4 9 10 16 15 20 21) ({n_rad} {n_arc} 1) simpleGrading ({g} 1 1)
    hex (0 1 4 5 11 12 15 16) ({n_rad} {n_arc} 1) simpleGrading ({g} 1 1)
    hex (1 2 3 4 12 13 14 15) ({n_x} {n_arc} 1) simpleGrading (1 1 1)
    hex (4 3 6 7 15 14 17 18) ({n_x} {n_y} 1) simpleGrading (1 1 1)
    hex (9 4 7 8 20 15 18 19) ({n_arc} {n_y} 1) simpleGrading (1 1 1)"""
    return verts, blocks, edges


def make_case(case, r, h, nu, ref=1):
    case = Path(case)
    subprocess.run(["rm", "-rf", str(case)], check=True)
    subprocess.run(["cp", "-r", TEMPLATE, str(case)], check=True)
    R, Hp = r * W, h * W
    verts, blocks, edges = block_mesh(R, Hp, ref)
    bm = case / "system/blockMeshDict"
    text = bm.read_text()
    text = re.sub(r"vertices\s*\(.*?\n\);", f"vertices\n(\n{verts}\n);",
                  text, flags=re.S)
    text = re.sub(r"blocks\s*\(.*?\n\);", f"blocks\n(\n{blocks}\n);",
                  text, flags=re.S)
    text = re.sub(r"edges\s*\(.*?\n\);", f"edges\n(\n{edges}\n);",
                  text, flags=re.S)
    bm.write_text(text)

    pp = case / "constant/physicalProperties"
    text = pp.read_text()
    text = re.sub(r"(nu\s*\{[^}]*value\s+)[\d.e+-]+", rf"\g<1>{nu}", text)
    pp.write_text(text)

    cd = case / "system/controlDict"
    text = cd.read_text()
    text = re.sub(r"endTime\s+\d+;", f"endTime         {ITER};", text)
    text = re.sub(r"writeInterval\s+\d+;", f"writeInterval   {ITER // 2};",
                  text)
    cd.write_text(text)

    gu = case / "system/graphUniform"
    text = gu.read_text()
    text = re.sub(r"start\s+\([^)]*\);", f"start           (0 {R} 0.25);", text)
    text = re.sub(r"end\s+\([^)]*\);", f"end             (0 {Hp} 0.25);", text)
    gu.write_text(text)
    # 오른쪽 끝 평균 변위 (patchAverage 함수 객체와 같은 설정)
    (case / "system/rightD").write_text(
        "patch           right;\n"
        "fields          (D);\n"
        "operation       areaAverage;\n"
        '#includeEtc "caseDicts/functions/surfaceFieldValue/patchValue.cfg"\n')


def sh(case, *cmd):
    log = Path(case) / f"log.{cmd[0]}.{len(list(Path(case).glob('log.*')))}"
    with open(log, "w") as f:
        subprocess.run(list(cmd), cwd=case, stdout=f, stderr=subprocess.STDOUT,
                       check=True)
    return log.read_text()


def extract(case):
    case = Path(case)
    times = sorted((p for p in case.iterdir() if p.name.isdigit()),
                   key=lambda p: int(p.name))
    out = {}
    for t in times[1:]:
        line = case / f"postProcessing/graphUniform/{t.name}/line.xy"
        row = next(r for r in line.read_text().splitlines()
                   if not r.startswith("#"))
        kt = float(row.split()[1]) / SIGMA0
        dat = next((case / "postProcessing/rightD").rglob("*.dat"))
        rows = [r for r in dat.read_text().splitlines()
                if r.strip() and not r.startswith("#")]
        d = {int(float(r.split()[0])): float(r.split()[1].strip("("))
             for r in rows}
        out[int(t.name)] = (kt, d[int(t.name)] * E / (SIGMA0 * W))
    return out


def run_plate(case, r, h, nu, ref=1):
    """케이스 하나 -> dict (Kt, C, 반복 중간값과의 차이, 셀 수, 시간)."""
    start = time.perf_counter()
    make_case(case, r, h, nu, ref)
    mesh = sh(case, "blockMesh")
    cells = int(re.search(r"nCells:\s*(\d+)", mesh).group(1))
    sh(case, "foamRun")
    sh(case, "foamPostProcess", "-func", "components(sigma)")
    sh(case, "foamPostProcess", "-func", "graphUniform")
    sh(case, "foamPostProcess", "-func", "rightD")
    res = extract(case)
    (kt_h, c_h), (kt, c) = res[ITER // 2], res[ITER]
    return {"r": r, "h": h, "nu": nu, "ref": ref, "cells": cells,
            "Kt": round(kt, 5), "C": round(c, 5),
            "dKt_half": round(abs(kt / kt_h - 1) * 100, 4),
            "seconds": round(time.perf_counter() - start, 2)}


if __name__ == "__main__":
    r, h, nu = (float(a) for a in sys.argv[1:4])
    case = Path(sys.argv[4] if len(sys.argv) > 4 else "plate").expanduser()
    ref = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    print(run_plate(case, r, h, nu, ref))
