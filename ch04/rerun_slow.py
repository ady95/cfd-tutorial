"""책 04-4: 수렴이 느렸을 뿐인 미수렴 케이스를 반복 상한을 늘려 다시 해석한다.

마지막 1000회 동안 최고 온도와 압력이 거의 변하지 않은(흔들림 0.5% 미만)
미수렴 케이스만 고른다. 값이 크게 오르내리는 케이스는 정상 상태 해가 없는
흐름이라 반복을 늘려도 수렴하지 않으므로 다시 돌리지 않는다.
결과는 작업 폴더의 rerun 폴더에 run_case 형식으로 남는다.

사용법: uv run python ch04/rerun_slow.py [작업 폴더]   (기본 ~/cfd-runs/dataset)
"""
import csv
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch03"))
import run_case as rc               # noqa: E402
from check_cases import swing       # noqa: E402

STABLE = 0.5            # 흔들림 기준 [%]
MAX_ITER = 20000        # 늘린 반복 상한

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/dataset")
run_dir = run_dir.expanduser()
rows = [r for r in csv.DictReader(open(run_dir / "results.csv"))
        if r["status"] == "not_converged"]
slow = [r for r in rows
        if max(swing(run_dir / r["case"]),
               swing(run_dir / r["case"], "pSection")) < STABLE]
print(f"미수렴 {len(rows)}개 가운데 값이 안정된 {len(slow)}개를 다시 해석")

# 반복 상한만 바꾼 템플릿을 만들어 run_case가 쓰게 한다
template = run_dir / "rerun" / "template"
if not template.exists():
    shutil.copytree(rc.TEMPLATE, template)
    ctrl = template / "system" / "controlDict"
    ctrl.write_text(re.sub(r"endTime\s+\d+;", f"endTime         {MAX_ITER};",
                           ctrl.read_text()))
rc.TEMPLATE = template
rc.TIMEOUT = 3600            # 반복을 4배로 늘렸으니 시간 제한도 늘린다

with ThreadPoolExecutor(6) as pool:
    jobs = [pool.submit(rc.run_case, float(r["H"]), float(r["e"]),
                        int(r["N"]), float(r["U"]), run_dir / "rerun")
            for r in slow]
    for job in jobs:
        r = job.result()
        print(f"{r['case']:<25} {r['status']:<13} 반복 {r.get('iterations')}"
              f"  {r['seconds']:6.1f}초  최고 {r.get('dT_max')}"
              f"  압력손실 {r.get('dp')}")
