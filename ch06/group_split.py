"""책 06-2: 나누는 방법에 따라 성능이 얼마나 달라지는지 교차 검증으로 비교한다.

1) cooling_v1(train+val)을 세 가지 방법으로 나눈다.
   - 무작위: 5겹으로 무작위로 나눈다.
   - 형상 기준: 리브 개수 N이 같은 케이스를 한 묶음으로 빼 둔다(N 5~14).
   - 운전조건 기준: 유속 U가 폭 0.02m/s 띠 안에 드는 케이스를 함께 빼 둔다.
   형상·운전조건 기준은 빼 두는 묶음을 범위 안쪽으로만 골라 외삽을 피한다.
2) 형상 하나를 유속 5개로 해석한 데이터(grouped_results.csv)를
   무작위 5겹과 형상 묶음 5겹으로 나눈다.
빼 둔 데이터의 예측을 모두 모아(out-of-fold) 상대 오차를 계산한다.

사용법: uv run python ch06/group_split.py   (CPU 6코어 기준 약 3분)
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold

from common import INPUTS, ROOT, TARGETS, fit_mlp, fit_poly, fit_xgb, load

sys.path.insert(0, str(ROOT / "ch04"))
from check_cases import judge       # noqa: E402  04-4의 판정 기준

MODELS = {"3차 다항": lambda a, b: fit_poly(a, b, 3),
          "6차 다항": lambda a, b: fit_poly(a, b, 6),
          "XGBoost": fit_xgb, "MLP": fit_mlp}


def run_fold(job):
    """한 묶음을 빼고 학습한 뒤, 빼 둔 묶음을 예측한다."""
    model, x, y, test = job
    torch.set_num_threads(1)            # 6개를 동시에 돌리므로 코어 하나씩
    train = np.setdiff1d(np.arange(len(x)), test)
    return MODELS[model](x[train], y[train])(x[test])


def width(s):
    return sum(2 if ord(c) > 0x1100 else 1 for c in s)   # 한글은 2칸


def compare(title, x, y, splits):
    """나누는 방법(splits)마다 모델별 out-of-fold 상대 오차를 출력한다."""
    jobs = [(m, x, y, te) for m in MODELS for folds in splits.values()
            for te in folds]
    with ProcessPoolExecutor(6) as pool:
        preds = iter(list(pool.map(run_fold, jobs)))
    print(f"\n[{title}] 데이터 {len(x)}개, 빼 두는 묶음: " + ", ".join(
        f"{k} {min(map(len, v))}~{max(map(len, v))}개"
        for k, v in splits.items()))
    print("평균 상대 오차 [%] dp/dT_max/dT_avg (괄호는 세 출력 중 최대)")
    print("모델      " + "".join(k + " " * (22 - width(k)) for k in splits))
    for model in MODELS:
        line = model + " " * (10 - width(model))
        for folds in splits.values():
            idx = np.concatenate(folds)
            p = np.concatenate([next(preds) for _ in folds])
            err = np.abs(p / y[idx] - 1) * 100
            cell = "/".join(f"{e:.2f}" for e in err.mean(axis=0))
            cell += f" ({err.max():.1f})"
            line += cell + " " * (22 - width(cell))
        print(line.rstrip())


start = time.perf_counter()

# 1) cooling_v1: 형상·운전조건이 모두 다른 LHS 데이터
x, y, df = load("train", "val")
N, U = df["N"].to_numpy(), df["U"].to_numpy()
compare("cooling_v1", x, y, {
    "무작위 5겹": [te for _, te in
                   KFold(5, shuffle=True, random_state=0).split(x)],
    "형상: N 하나씩": [np.where(N == n)[0] for n in range(5, 15)],
    "운전: U 띠": [np.where((U >= lo) & (U < lo + 0.02))[0]
                   for lo in (0.07, 0.09, 0.11)],
})

# 2) 형상 하나를 유속 5개로 해석한 데이터: 04-4 판정을 통과한 케이스만
g = pd.read_csv(ROOT / "ch06" / "grouped_results.csv")
g = g[[judge(r)[0] for r in g.to_dict("records")]].reset_index(drop=True)
xg, yg = g[INPUTS].to_numpy(float), g[TARGETS].to_numpy(float)
compare("형상 30개 x 유속 5개", xg, yg, {
    "무작위 5겹": [te for _, te in
                   KFold(5, shuffle=True, random_state=0).split(xg)],
    "형상 묶음 5겹": [te for _, te in
                      GroupKFold(5).split(xg, groups=g["split"])],
})
print(f"\n전체 {time.perf_counter() - start:.0f}초")
