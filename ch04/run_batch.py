"""책 04-3: samples.csv의 설계 변수를 여러 개 동시에 해석한다.

케이스마다 코어 하나를 쓰는 ch03/run_case.py를 동시에 workers개씩 돌린다.
케이스가 끝날 때마다 진행 상황을 한 줄 출력하고 작업 폴더의 results.csv를
다시 쓴다. Ctrl+C로 멈추면 돌던 케이스가 끝나거나 멈출 때까지 기다린 뒤
끝내며, 같은 명령을 다시 실행하면 끝나지 않은 케이스만 이어서 돌린다.

사용법: uv run python ch04/run_batch.py [작업 폴더] [--workers 6]
                                          [--split train,val] [--limit 24]
    작업 폴더 기본값 ~/cfd-runs/dataset
"""
import argparse
import csv
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch03"))
from env_info import record_env     # noqa: E402
from run_case import run_case       # noqa: E402

COLUMNS = ["id", "split", "case", "H", "e", "N", "U", "e_ratio", "status",
           "error", "cells", "iterations", "dp", "dT_max", "dT_avg",
           "dT_out", "dT_out_energy", "seconds"]


def save(run_dir, done):
    """끝난 케이스를 id 순서로 results.csv에 쓴다."""
    with open(run_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(done.values(), key=lambda r: int(r["id"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="~/cfd-runs/dataset")
    ap.add_argument("--workers", type=int, default=6, help="동시 해석 수")
    ap.add_argument("--split", default="", help="예: train,val (기본 전체)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 n개만")
    args = ap.parse_args()
    if shutil.which("foamRun") is None:
        sys.exit("OpenFOAM 환경이 없습니다: . /opt/openfoam14/etc/bashrc")

    run_dir = Path(args.run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    changed = record_env(run_dir)
    if changed:
        print("주의: 처음 실행할 때와 환경이 다릅니다:", ", ".join(changed))
    with open(HERE / "samples.csv", newline="") as f:
        samples = list(csv.DictReader(f))
    if args.split:
        samples = [s for s in samples if s["split"] in args.split.split(",")]
    if args.limit:
        samples = samples[:args.limit]

    n, done, start = len(samples), {}, time.time()
    print(f"{n}개 해석, 동시 {args.workers}개 -> {run_dir}")
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(run_case, float(s["H"]), float(s["e"]),
                            int(s["N"]), float(s["U"]), run_dir): s
                for s in samples}
        try:
            for k, job in enumerate(as_completed(jobs), 1):
                r = {**jobs[job], **job.result()}
                done[r["id"]] = r
                save(run_dir, done)
                left = (time.time() - start) / k * (n - k) / 60
                print(f"[{k:3d}/{n}] {r['case']:<25} {r['status']:<13}"
                      f"{r['seconds']:6.1f}초  남은 약 {left:.0f}분",
                      flush=True)
        except KeyboardInterrupt:
            print("중단합니다. 돌던 케이스가 끝나기를 기다립니다.")
            pool.shutdown(wait=True, cancel_futures=True)
            print(f"{len(done)}개 저장됨. 같은 명령으로 이어서 돌립니다.")
            sys.exit(1)

    count = {s: sum(r["status"] == s for r in done.values())
             for s in ("ok", "not_converged", "failed")}
    sec = time.time() - start
    print(f"완료 {count['ok']}개, 미수렴 {count['not_converged']}개,"
          f" 실패 {count['failed']}개, 걸린 시간 {sec / 60:.1f}분 ({sec:.0f}초)")
    print(f"결과: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
