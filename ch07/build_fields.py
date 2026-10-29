"""책 07-3: 사용 가능한 케이스 전체를 균일 격자 배열 파일 하나로 저장한다.

cooling_v1에서 valid가 True인 690개(train·val·test·ood_U·ood_H)를 1025 x 49
격자로 옮긴다. 나누기는 데이터셋의 split 열을 그대로 따른다.

사용법: uv run python ch07/build_fields.py [해석 폴더] [출력 파일]
    기본값  ~/cfd-runs/dataset   ~/cfd-runs/ch07/fields_1025x49.npz
        CPU 6코어 기준 약 1분 10초, 파일 약 0.7 GB
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grid import FIELDS, scalars, to_grid
from resolution import DATA, case_dir

NX, NY = 1025, 49


def one(args):
    path, H, e, N = args
    return to_grid(path, H, e, N, NX, NY)


if __name__ == "__main__":
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/dataset")
    out = Path(sys.argv[2] if len(sys.argv) > 2
               else f"~/cfd-runs/ch07/fields_{NX}x{NY}.npz")
    run, out = run.expanduser(), out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)
    df = df[df["valid"]].reset_index(drop=True)
    jobs = [(case_dir(run, r), r.H, r.e, r.N) for r in df.itertuples()]
    start = time.perf_counter()
    with ProcessPoolExecutor(6) as pool:
        res = list(pool.map(one, jobs, chunksize=4))
    mask = np.stack([m for m, _ in res]).astype(np.uint8)
    fields = np.stack([f for _, f in res])
    grid_scalars = np.array([scalars(m, f) for m, f in res])
    np.savez(out, mask=mask, fields=fields, field_names=FIELDS,
             case=df["case"].to_numpy(str), split=df["split"].to_numpy(str),
             inputs=df[["H", "e_ratio", "N", "U"]].to_numpy(np.float32),
             cfd_scalars=df[["dp", "dT_max", "dT_avg"]].to_numpy(),
             grid_scalars=grid_scalars)
    print(f"{len(df)}개 변환 {time.perf_counter() - start:.0f}초 -> {out}")
    print(f"mask {mask.shape} {mask.dtype}, fields {fields.shape} "
          f"{fields.dtype}, 파일 {out.stat().st_size / 1e9:.2f} GB")
    print("split별 개수:", df["split"].value_counts().to_dict())
