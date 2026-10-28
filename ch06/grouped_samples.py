"""책 06-2: 형상 하나를 여러 유속으로 해석하는 '묶음' 샘플 목록을 만든다.

실무 데이터는 흔히 이런 모양이다. 시제품 형상 몇 개를 정해 두고 운전조건만
바꿔 가며 해석하거나 시험한다. 형상 30개를 LHS로 뽑고, 형상마다 유속 5개를
붙여 150개 케이스를 만든다. 04-2의 cooling_v1과 겹치지 않도록 시드를 바꾼다.

사용법: uv run python ch06/grouped_samples.py   (결과 ch06/grouped_samples.csv)
해석:   uv run python ch04/run_batch.py ~/cfd-runs/grouped \\
            --samples ch06/grouped_samples.csv
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch04"))
from make_samples import RANGES, invalid, lhs    # noqa: E402

SEED = 2606
N_GEOM = 30
U_LEVELS = [0.05, 0.075, 0.1, 0.125, 0.15]

shapes = [(H, e, N) for H, e, N, _ in lhs(N_GEOM, RANGES, SEED)]
out = HERE / "grouped_samples.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["id", "split", "H", "e", "N", "U", "e_ratio"])
    i = 0
    for g, (H, e, N) in enumerate(shapes):
        for U in U_LEVELS:
            assert invalid(H, e, N, U) is None
            w.writerow([i, f"g{g:02d}", H, e, N, U, round(e / H, 4)])
            i += 1
print(f"{out} 에 형상 {N_GEOM}개 x 유속 {len(U_LEVELS)}개 = {i}개 저장")
