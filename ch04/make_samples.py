"""책 04-2: 대리모델 학습용 설계 변수 샘플 목록을 만든다.

설계 범위 안에서 LHS로 600개를 뽑아 학습·검증·시험용으로 나누고,
범위 바깥(OOD) 시험용으로 유속이 빠른 쪽과 채널이 높은 쪽을 50개씩 더 뽑는다.
난수 시드가 고정되어 있어 언제 실행해도 같은 목록이 나온다.

사용법: uv run python ch04/make_samples.py [출력 파일]   (기본 ch04/samples.csv)
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

SEED = 2026
# 설계 범위 (03-1, 03-2에서 정함). 리브 높이는 채널 높이에 대한 비율로 뽑는다
RANGES = {"H": (1.5, 2.5), "e_ratio": (0.2, 0.4), "N": (4, 15),
          "U": (0.05, 0.15)}
N_IN = 600
SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
# 범위 바깥 시험용: 한 변수만 범위를 넘기고 나머지는 설계 범위 그대로
OOD = {"ood_U": ("U", (0.15, 0.20), 50), "ood_H": ("H", (2.5, 3.0), 50)}

L_HEAT, W_RIB = 60.0, 0.5   # 가열 구간 길이, 리브 폭 [mm] (channel_mesh.py)


def lhs(n, ranges, seed):
    """범위 dict대로 LHS 샘플을 뽑아 반올림한 (H, e, N, U) 목록을 돌려준다."""
    u = qmc.LatinHypercube(d=4, seed=seed).random(n)
    (h0, h1), (r0, r1), (n0, n1), (u0, u1) = ranges.values()
    rows = []
    for a, b, c, d in u:
        H = round(h0 + a * (h1 - h0), 3)          # 0.001 mm 단위
        e = round((r0 + b * (r1 - r0)) * H, 3)
        N = int(n0 + c * (n1 - n0 + 1))           # 정수 n0~n1이 같은 확률
        U = round(u0 + d * (u1 - u0), 4)          # 0.0001 m/s 단위
        rows.append((H, e, N, U))
    return rows


def invalid(H, e, N, U):
    """형상으로 만들 수 없는 조합이면 이유를, 괜찮으면 None을 돌려준다."""
    if not 0 < e < H:
        return "리브가 채널보다 높음"
    if L_HEAT / N <= 2 * W_RIB:
        return "리브 사이 간격이 리브 폭의 2배 이하"
    if U <= 0:
        return "유속이 0 이하"
    return None


def main(out):
    rng = np.random.default_rng(SEED)
    samples = []
    rows = lhs(N_IN, RANGES, SEED)
    order = rng.permutation(N_IN)                 # 무작위로 섞어 나눈다
    cut = np.cumsum([int(N_IN * f) for f in SPLIT.values()])
    split = np.empty(N_IN, dtype=object)
    for name, idx in zip(SPLIT, np.split(order, cut[:-1])):
        split[idx] = name
    samples += [(s, *r) for s, r in zip(split, rows)]
    for k, (name, (var, rng_ood, n)) in enumerate(OOD.items(), 1):
        ranges = dict(RANGES, **{var: rng_ood})
        samples += [(name, *r) for r in lhs(n, ranges, SEED + k)]

    bad = [(s, r) for s, *r in samples if invalid(*r)]
    if bad:
        sys.exit(f"만들 수 없는 조합 {len(bad)}개: {bad[:3]}")
    names = {f"H{H:g}_e{e:g}_N{N}_U{U:g}" for _, H, e, N, U in samples}
    assert len(names) == len(samples), "반올림 뒤 겹치는 조합이 있음"

    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "split", "H", "e", "N", "U", "e_ratio"])
        for i, (s, H, e, N, U) in enumerate(samples):
            w.writerow([i, s, H, e, N, U, round(e / H, 4)])

    print(f"{out} 에 {len(samples)}개 저장 (시드 {SEED})")
    print("구분       개수   H [mm]       e/H          N      U [m/s]")
    for s in [*SPLIT, *OOD]:
        x = np.array([(H, e / H, N, U) for t, H, e, N, U in samples
                      if t == s])
        lo, hi = x.min(axis=0), x.max(axis=0)
        print(f"{s:<9} {len(x):4d}   {lo[0]:.2f}~{hi[0]:.2f}   "
              f"{lo[1]:.3f}~{hi[1]:.3f}  {lo[2]:2.0f}~{hi[2]:2.0f}  "
              f"{lo[3]:.3f}~{hi[3]:.3f}")


if __name__ == "__main__":
    default = Path(__file__).resolve().parent / "samples.csv"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
