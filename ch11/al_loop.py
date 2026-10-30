"""책 11-3, 11-4: 능동학습 반복 루프 (학습 -> 평가 -> 선택 -> CFD -> 추가).

train에서 무작위로 50개를 골라 시작하고, 한 번에 25개씩 10번 CFD를 더한다.
선택 방법(--strategy)만 바꿔 같은 후보 풀에서 비교한다.
  random   무작위 (비교군)
  top      불확실성 상위
  diverse  불확실성 + 다양성 (수렴하지 않은 설계 근처는 피함)
  manual   06-5에서 사람이 고른 구간(모서리, 리브 4~5개, 범위 끝)에서 무작위
반복마다 앙상블을 처음부터 다시 학습해 새 시험 데이터 200개와 범위 밖
데이터로 평가하고, 기록을 작업 폴더의 history.json에 남긴다.
CFD는 04-3의 run_batch.py로 돌리므로 중간에 멈춰도 같은 명령으로 이어진다.

--space wide이면 후보를 넓힌 범위(e/H 0.5, 리브 20개까지)의 새 영역에서
고르고, 넓힌 범위의 시험 데이터(wide_test.csv)로도 평가한다 (11-5).

사용법: uv run python ch11/al_loop.py --strategy diverse --seed 0
          [--n0 50] [--rounds 10] [--batch 25] [--space base] [--workers 2]
    작업 폴더 ~/cfd-runs/ch11/al/<space>_n<n0>/<strategy>_s<seed>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (RANGES, WIDE, corner, fit_ensemble, load, pool,
                    rows_to_arrays, run_cfd, summary, to_unit, wide_pool)
from new_test import load_new_test, load_wide_test
from select_samples import describe, pick_diverse, pick_random, pick_top


def pick_manual(x, k, rng):
    """06-5의 세 구간(모서리, 리브 4~5개, 범위 끝)에서 같은 수만큼 무작위로."""
    xu = to_unit(x)
    edge = ((xu[:, [0, 1, 3]] < 0.1) | (xu[:, [0, 1, 3]] > 0.9)).any(axis=1)
    edge |= np.isin(x[:, 2], [4, 15])
    chosen = []
    for j, region in enumerate([corner(x), x[:, 2] <= 5, edge]):
        m = k // 3 + (j < k % 3)
        free = np.setdiff1d(np.flatnonzero(region), chosen)
        chosen += list(rng.choice(free, m, replace=False))
    return np.array(chosen)


def errors(ens, x, y):
    """세 출력 상대 오차의 평균과 최대 [%]."""
    m, _ = ens.predict(x)
    e = np.abs(m / y - 1) * 100
    return e.mean(), e.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True,
                    choices=["random", "top", "diverse", "manual"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n0", type=int, default=50, help="처음 데이터 수")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--space", choices=["base", "wide"], default="base")
    ap.add_argument("--workers", type=int, default=2, help="CFD 동시 해석 수")
    args = ap.parse_args()
    torch.set_num_threads(1)
    run_dir = (Path("~/cfd-runs/ch11/al").expanduser()
               / f"{args.space}_n{args.n0}" / f"{args.strategy}_s{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    x_tr, y_tr, _ = load("train")
    idx = rng.permutation(len(x_tr))[:args.n0]
    x, y = x_tr[idx], y_tr[idx]
    x_test, y_test = load_new_test()
    x_ood, y_ood, _ = load("ood_U", "ood_H")
    c_test = corner(x_test)
    wide = args.space == "wide"
    cand, ranges = (wide_pool(), WIDE) if wide else (pool(), RANGES)
    if wide:
        x_wide, y_wide, _ = load_wide_test()
    left = np.ones(len(cand), bool)     # 아직 고르지 않은 후보
    failed = np.empty((0, 4))
    hist, cfd_sec, start = [], 0.0, time.time()

    for r in range(args.rounds + 1):
        t = time.time()
        ens = fit_ensemble(x, y, seed=args.seed)
        train_sec = time.time() - t
        mean, worst = errors(ens, x_test, y_test)
        rec = {"round": r, "n": len(x), "failed": len(failed),
               "test_mean": mean, "test_max": worst,
               "corner_mean": errors(ens, x_test[c_test], y_test[c_test])[0],
               "ood_mean": errors(ens, x_ood, y_ood)[0],
               "wide_mean": errors(ens, x_wide, y_wide)[0] if wide else None,
               "wide_max": errors(ens, x_wide, y_wide)[1] if wide else None,
               "cfd_seconds": cfd_sec, "train_seconds": train_sec}
        # 이 반복을 시작할 때의 상태: 데이터 수, 오차, 지금까지의 미수렴과 CFD
        print(f"[{args.strategy} s{args.seed}] {r:2d}회 {len(x):3d}개"
              f"  평균 {mean:.3f}% 최대 {worst:.2f}%"
              + (f"  넓힌 범위 {rec['wide_mean']:.3f}%" if wide else "")
              + f"  미수렴 {len(failed)}  CFD {cfd_sec / 60:.1f}분"
              f"  경과 {(time.time() - start) / 60:.0f}분", flush=True)
        if r < args.rounds:
            free = np.flatnonzero(left)
            if args.strategy == "random":
                i = pick_random(len(free), args.batch, rng)
            elif args.strategy == "manual":
                i = pick_manual(cand[free], args.batch, rng)
            else:
                u = summary(ens.predict(cand[free])[1])
                i = (pick_top(u, args.batch) if args.strategy == "top" else
                     pick_diverse(u, to_unit(cand[free], ranges), args.batch,
                                  avoid=to_unit(failed, ranges)))
            chosen = free[i]
            left[chosen] = False
            rows = run_cfd(cand[chosen], run_dir, workers=args.workers,
                           split=f"round{r + 1}")
            xn, yn, bad = rows_to_arrays(rows)
            x, y = np.vstack([x, xn]), np.vstack([y, yn])
            failed = np.vstack([failed, bad])
            sec = [float(row["seconds"]) for row in rows]
            fail_sec = sum(s for s, row in zip(sec, rows)
                           if row["status"] != "ok")
            cfd_sec += sum(sec)
            rec.update(added=len(xn), new_failed=len(bad),
                       fail_seconds=fail_sec, picked=describe(cand[chosen]),
                       chosen=cand[chosen].tolist())
        hist.append(rec)
        (run_dir / "history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
