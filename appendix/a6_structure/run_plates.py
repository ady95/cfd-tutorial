"""책 부록 A-6: 평판 인장 해석 데이터셋을 LHS로 만든다 (4장과 같은 흐름).

설계 범위: 구멍 반지름 비 r 0.1~0.4, 판 높이 비 h 1.0~2.0, 푸아송비 0.25~0.35
- train 100, val 25, test 25 (LHS 150점, 시드 2026)
- ood_r 10: 구멍이 더 큰 쪽 r 0.4~0.45 (나머지는 범위 안)
격자는 ref 4 (부록 A-6 본문의 격자 검사). 동시 6개.

사용법: python3 appendix/a6_structure/run_plates.py [작업 폴더] [결과 CSV]
    기본값  ~/cfd-runs/a6/plates   appendix/a6_structure/plates.csv
"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_case import run_plate  # noqa: E402

SEED = 2026
LO, HI = np.array([0.1, 1.0, 0.25]), np.array([0.4, 2.0, 0.35])
REF = 4


def samples():
    rng = np.random.default_rng(SEED)
    x = qmc.scale(qmc.LatinHypercube(d=3, seed=rng).random(150), LO, HI)
    split = ["train"] * 100 + ["val"] * 25 + ["test"] * 25
    ood = qmc.scale(qmc.LatinHypercube(d=3, seed=rng).random(10),
                    [0.4, 1.0, 0.25], [0.45, 2.0, 0.35])
    rows = [(s, *v) for s, v in zip(split, x)]
    rows += [("ood_r", *v) for v in ood]
    return [{"id": i, "split": s, "r": round(r, 4), "h": round(h, 4),
             "nu": round(nu, 4)} for i, (s, r, h, nu) in enumerate(rows)]


def one(args):
    row, work = args
    res = run_plate(work / f"p{row['id']:03d}", row["r"], row["h"],
                    row["nu"], REF)
    return {**row, **{k: res[k] for k in ("cells", "Kt", "C", "dKt_half",
                                          "seconds")}}


if __name__ == "__main__":
    work = Path(sys.argv[1] if len(sys.argv) > 1
                else "~/cfd-runs/a6/plates").expanduser()
    out = Path(sys.argv[2] if len(sys.argv) > 2 else
               Path(__file__).resolve().parent / "plates.csv")
    work.mkdir(parents=True, exist_ok=True)
    rows = samples()
    start = time.perf_counter()
    with ProcessPoolExecutor(6) as pool:
        res = list(pool.map(one, [(r, work) for r in rows]))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(res)
    sec = np.array([r["seconds"] for r in res])
    print(f"{len(res)}개 해석, 동시 6개, 걸린 시간 "
          f"{(time.perf_counter() - start) / 60:.1f}분 -> {out}")
    print(f"케이스 하나 {sec.min():.1f}~{sec.max():.1f}초 (중앙값 "
          f"{np.median(sec):.1f}초), 셀 "
          f"{min(r['cells'] for r in res):,}~{max(r['cells'] for r in res):,}개")
    print(f"반복 500 -> 1000 사이 Kt 변화 최대 "
          f"{max(r['dKt_half'] for r in res):.4f}%")
    print(f"Kt {min(r['Kt'] for r in res):.3f}~"
          f"{max(r['Kt'] for r in res):.3f}, C {min(r['C'] for r in res):.3f}"
          f"~{max(r['C'] for r in res):.3f}")
    print(f"작업 폴더 크기: {os.popen(f'du -sh {work}').read().split()[0]}")
