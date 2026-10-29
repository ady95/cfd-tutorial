"""책 07-3: 균일 격자의 해상도를 정한다.

해상도 세 가지로 케이스를 격자에 옮긴 뒤, 격자 배열에서 다시 계산한
dp, dT_max, dT_avg가 3장 함수 객체 값(데이터셋 값)과 얼마나 다른지 본다.
옮기는 과정에서 잃는 정보가 이 차이다.

사용법: uv run python ch07/resolution.py [해석 폴더]   (기본 ~/cfd-runs/dataset)
        train에서 60개를 골라 비교한다. CPU 6코어 기준 약 1분
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid import scalars, to_grid

DATA = Path(__file__).resolve().parent.parent / "data/cooling_v1/cases.parquet"
SIZES = [(257, 25), (513, 49), (1025, 49), (1025, 97)]   # (nx, ny)
TARGETS = ["dp", "dT_max", "dT_avg"]


def case_dir(run, row):
    """재해석한 케이스는 rerun 폴더의 결과를 쓴다 (04-4)."""
    return run / "rerun" / row.case if row.rerun else run / row.case


def one(args):
    path, (H, e, N), nx, ny = args
    start = time.perf_counter()
    mask, fields = to_grid(path, H, e, N, nx, ny)
    return scalars(mask, fields), time.perf_counter() - start


if __name__ == "__main__":
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/dataset")
    run = run.expanduser()
    df = pd.read_parquet(DATA)
    df = df[df["valid"] & (df["split"] == "train")].sample(60, random_state=0)
    truth = df[TARGETS].to_numpy()
    print("격자 (nx x ny)  간격 dx/dy [mm]  케이스당  "
          "상대 차이 평균/최대 [%]")
    print(" " * 43 + "dp       dT_max     dT_avg")
    for nx, ny in SIZES:
        jobs = [(case_dir(run, r), (r.H, r.e, r.N), nx, ny)
                for r in df.itertuples()]
        with ProcessPoolExecutor(6) as pool:
            res = list(pool.map(one, jobs))
        got = np.array([s for s, _ in res])
        sec = np.mean([t for _, t in res])
        err = np.abs(got / truth - 1) * 100
        text = "  ".join(f"{m:.2f}/{x:.2f}"
                         for m, x in zip(err.mean(0), err.max(0)))
        print(f"{nx:5d} x {ny:3d}   {80 / (nx - 1):.4f}/{3 / (ny - 1):.4f}"
              f"  {sec:6.2f}초   {text}")
