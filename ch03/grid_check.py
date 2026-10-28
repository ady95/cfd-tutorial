"""책 03-5: 리브 채널의 격자 독립성 검사.

설계 변수 범위 끝의 형상·유속 다섯 개를 채널 높이 방향 셀 15, 30, 60개
격자로 해석하고, 셀 30(이 책의 기본 격자)이 셀 60과 얼마나 다른지 본다.

사용법: uv run python ch03/grid_check.py [작업 폴더]   (기본 ~/cfd-runs/gridrib)
이미 해석한 케이스는 건너뛰므로 중간에 멈춰도 같은 명령으로 이어 돌린다.
"""
import shutil
import sys
from pathlib import Path

import channel_mesh
from run_case import run_case

SHAPES = [(1.5, 0.3, 4, 0.05), (1.5, 0.6, 15, 0.1), (2, 0.6, 8, 0.1),
          (2.5, 0.5, 15, 0.15), (2.5, 1.0, 4, 0.05)]
GRIDS = [15, 30, 60]            # 채널 높이 방향 셀 수
VALUES = [("dp", "압력손실", "{:.5f}"), ("dT_max", "최고 온도 상승", "{:.3f}"),
          ("dT_avg", "평균 온도 상승", "{:.3f}")]

if shutil.which("foamRun") is None:
    sys.exit("OpenFOAM 환경이 없습니다: . /opt/openfoam14/etc/bashrc")
run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/gridrib")

res = {}
for nh in GRIDS:
    channel_mesh.CELLS_H = nh   # block_mesh_dict가 이 값을 읽어 격자를 만든다
    for s in SHAPES:
        r = run_case(*s, run_dir.expanduser() / f"nh{nh}")
        res[s, nh] = r
        print(f"셀 {nh:2d}  {r['case']:<19} {r['status']:<13}"
              f" 반복 {r.get('iterations') or '-':>4}  {r['seconds']:6.1f}초",
              flush=True)

for key, label, fmt in VALUES:
    print(f"\n{label}      셀 15     셀 30     셀 60   30->60 변화")
    for s in SHAPES:
        rs = [res[s, nh] for nh in GRIDS]
        if any(r["status"] != "ok" for r in rs):
            print(f"{rs[0]['case']:<19} 수렴하지 않은 격자가 있음")
            continue
        v = [r[key] for r in rs]
        print(f"{rs[0]['case']:<19}" + "".join(f"{fmt.format(x):>10}"
                                               for x in v)
              + f"   {(v[2] / v[1] - 1) * 100:+6.2f}%")
