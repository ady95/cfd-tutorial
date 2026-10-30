"""책 부록 A-6: 평판 해석의 격자 검사.

구멍 크기 세 가지(r 0.1, 0.25, 0.4)를 격자 4단계(ref 1, 2, 4, 8)로 풀고,
ref 8과 ref 4로 외삽한 값(1차 수렴 가정: 2 x K8 - K4)과 비교한다.
r 0.1은 판을 높게(h 2.0) 두어 무한 평판에 가까운 공학 공식과도 비교한다.

사용법: python3 appendix/a6_structure/grid_check.py [작업 폴더]
    기본값 ~/cfd-runs/a6/grid   (동시 6개, 약 3분)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_case import run_plate  # noqa: E402

CASES = [(0.1, 2.0), (0.25, 1.0), (0.4, 1.0)]
REFS = [1, 2, 4, 8]


def one(args):
    work, r, h, ref = args
    return run_plate(work / f"r{r}_h{h}_ref{ref}", r, h, 0.3, ref)


if __name__ == "__main__":
    work = Path(sys.argv[1] if len(sys.argv) > 1
                else "~/cfd-runs/a6/grid").expanduser()
    work.mkdir(parents=True, exist_ok=True)
    jobs = [(work, r, h, ref) for r, h in CASES for ref in REFS]
    with ProcessPoolExecutor(6) as pool:
        res = list(pool.map(one, jobs))
    print("r     h    ref   셀 수     Kt      외삽 대비     C")
    for r, h in CASES:
        rows = [x for x in res if x["r"] == r]
        k8 = next(x["Kt"] for x in rows if x["ref"] == 8)
        k4 = next(x["Kt"] for x in rows if x["ref"] == 4)
        ext = 2 * k8 - k4
        for x in rows:
            print(f"{r:<5} {h:<4} {x['ref']:>3} {x['cells']:>7,}  "
                  f"{x['Kt']:.4f}  {(x['Kt'] / ext - 1) * 100:+6.2f}%   "
                  f"{x['C']:.4f}")
        print(f"{'':<14}외삽     {ext:.4f}")
    heywood = (2 + 0.9 ** 3) / 0.9
    print(f"\nr 0.1의 Heywood 공식 값 {heywood:.4f}")
