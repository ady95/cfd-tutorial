"""책 11-1, 11-5: 11장 평가에 쓸 새 시험 데이터를 만들고 CFD로 해석한다.

06-1에서 test 결과를 보고 모델을 고치지 않기로 했으므로, 데이터를 더해
개선한 모델은 처음 보는 데이터로 평가한다. 04-2와 같은 LHS로 시드만 바꿔
설계 범위 안에서 200개를 뽑고, 04-3의 run_batch.py로 해석해
ch11/new_test.csv에 저장한다 (저장소에 넣어 두었으므로 다시 돌릴 필요 없음).

--wide이면 넓힌 범위(e/H 0.5, 리브 20개까지)에서 원래 범위 밖이고 틈 제약을
만족하는 점 100개를 같은 방법으로 만들어 ch11/wide_test.csv에 저장한다 (11-5).

사용법: uv run python ch11/new_test.py [--wide] [작업 폴더]
    작업 폴더 기본값 ~/cfd-runs/ch11/new_test (--wide이면 wide_test)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from common import GAP_MIN, WIDE, outside, rows_to_arrays, run_cfd, to_design

HERE = Path(__file__).resolve().parent
SEED, N_NEW, N_WIDE = 2027, 200, 100
KEYS = ["H", "e_ratio", "N", "U", "status", "iterations", "dp", "dT_max",
        "dT_avg", "seconds"]


def load_csv(path):
    """수렴한 케이스의 (x, y)와 수렴하지 않은 케이스의 x."""
    with open(path, newline="") as f:
        return rows_to_arrays(list(csv.DictReader(f)))


def load_new_test():
    """범위 안 새 시험 데이터 (x, y)."""
    return load_csv(HERE / "new_test.csv")[:2]


def load_wide_test():
    """넓힌 범위 새 영역의 시험 데이터 (x, y, 수렴하지 않은 x)."""
    return load_csv(HERE / "wide_test.csv")


def samples(wide):
    if not wide:
        return to_design(qmc.LatinHypercube(d=4, seed=SEED).random(N_NEW))
    # 넓힌 범위 전체에서 뽑아 새 영역이면서 틈 제약을 만족하는 점만 남긴다
    x = to_design(qmc.LatinHypercube(d=4, seed=SEED + 1).random(1000), WIDE)
    x = x[outside(x) & (x[:, 0] * (1 - x[:, 1]) >= GAP_MIN)]
    return x[:N_WIDE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--wide", action="store_true")
    args = ap.parse_args()
    name = "wide_test" if args.wide else "new_test"
    run_dir = Path(args.run_dir or f"~/cfd-runs/ch11/{name}").expanduser()
    out = HERE / f"{name}.csv"

    x = samples(args.wide)
    print(f"LHS {len(x)}개 해석 -> {run_dir}")
    rows = run_cfd(x, run_dir, split=name)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, KEYS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    sec = np.array([float(r["seconds"]) for r in rows])
    status = [r["status"] for r in rows]
    print(f"수렴 {status.count('ok')}개, 미수렴 {status.count('not_converged')}개,"
          f" 실패 {status.count('failed')}개, 케이스 시간 합 {sec.sum() / 60:.1f}분")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
