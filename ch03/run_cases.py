"""책 03-4: CSV 파일에 적힌 설계 변수 목록을 차례로 해석한다.

사용법: uv run python ch03/run_cases.py 변수.csv [작업 폴더]
    예: uv run python ch03/run_cases.py ch03/demo_cases.csv ~/cfd-runs/auto
변수 CSV는 H,e,N,U 네 열이다. 해석이 끝날 때마다 작업 폴더의 results.csv를
다시 쓰므로, 중간에 멈춰도 그때까지의 결과가 남는다. 같은 명령을 다시
실행하면 result.json이 있는 케이스는 건너뛴다.
"""
import csv
import shutil
import sys
from pathlib import Path

from run_case import run_case

COLUMNS = ["case", "H", "e", "N", "U", "status", "error", "cells",
           "iterations", "dp", "dT_max", "dT_avg", "dT_out",
           "dT_out_energy", "seconds"]


def main(param_csv, run_dir):
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(param_csv, newline="") as f:
        params = list(csv.DictReader(f))

    records = []
    for i, p in enumerate(params, 1):
        r = run_case(float(p["H"]), float(p["e"]), int(p["N"]),
                     float(p["U"]), run_dir)
        records.append(r)
        print(f"[{i}/{len(params)}] {r['case']:<22} {r['status']:<13}"
              f" {r['seconds']:6.1f}초  {r['error'] or ''}", flush=True)
        with open(run_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(records)

    count = {s: sum(r["status"] == s for r in records)
             for s in ("ok", "not_converged", "failed")}
    print(f"완료 {count['ok']}개, 미수렴 {count['not_converged']}개,"
          f" 실패 {count['failed']}개 -> {run_dir / 'results.csv'}")


if __name__ == "__main__":
    if shutil.which("foamRun") is None:
        sys.exit("OpenFOAM 환경이 없습니다: . /opt/openfoam14/etc/bashrc")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "~/cfd-runs/auto")
