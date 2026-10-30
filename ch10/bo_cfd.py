"""책 10-5: 대리모델 없이 CFD를 직접 BoTorch로 최적화하고 비용을 비교한다.

10-3과 같은 문제(펌핑 동력 ≤ 기준 설계, 틈 ≥ 1.0mm에서 dT_max 최소화)를
평가 하나마다 CFD(셀 30, 코어 1개)를 돌려 푼다. 수렴하지 않은 설계는 실패로
기록하고 GP 학습에서 뺀다.

사용법: uv run python ch10/bo_cfd.py [--budget 60] [--seed 0]
        시드마다 코어 하나를 쓰므로 시드 여러 개를 동시에 돌려도 된다
결과:   진행 출력, ~/cfd-runs/ch10/bo_cfd/s<시드>/history.json
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bo import bo_minimize
from problem import BASE, NAMES, cfd

OUT = Path("~/cfd-runs/ch10/bo_cfd").expanduser()


def make_evaluate(run_dir, p_max, history):
    """설계 (n, 4) -> CFD의 dT_max (n,), 펌핑 동력 제약 값 (n, 1)."""
    def evaluate(x):
        rs = cfd(x, run_dir, workers=1)
        history.extend({**dict(zip(NAMES, map(float, v))), **r}
                       for v, r in zip(x, rs))
        f = np.array([r["dT_max"] if r["dT_max"] else np.nan for r in rs])
        c = np.array([[r["P"] / p_max - 1 if r["P"] else np.nan]
                      for r in rs])
        return f, c
    return evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if shutil.which("foamRun") is None:
        sys.exit("OpenFOAM 환경이 없습니다: . /opt/openfoam14/etc/bashrc")
    torch.set_num_threads(1)
    run_dir = OUT / f"s{args.seed}"
    p_max = cfd(BASE, run_dir, workers=1)[0]["P"]     # 기준 설계도 CFD로
    history = []

    start = time.time()
    f, c, x = bo_minimize(make_evaluate(run_dir, p_max, history),
                          args.budget, seed=args.seed, verbose=True)
    sec = time.time() - start
    feas = np.isfinite(f) & (c[:, 0] <= 0)
    runs = {h["case"]: h["seconds"] for h in history}  # 같은 설계는 한 번만 해석
    print(f"\n평가 {len(f)}회 (해석 {len(runs)}건), 걸린 시간 {sec / 60:.1f}분,"
          f" 그중 CFD {sum(runs.values()) / 60:.1f}분")
    print(f"실패 {int((~np.isfinite(f)).sum())}개, "
          f"제약 위반 {int((np.isfinite(f) & ~feas).sum())}개")
    best = np.minimum.accumulate(np.where(feas, f, np.inf))
    print("찾은 최소 dT_max  " + "  ".join(
        f"{n}회 {best[n - 1]:.4f}" for n in (20, 40, len(f))))
    i = np.flatnonzero(feas)[np.argmin(f[feas])]
    print("최적  " + "  ".join(f"{n} {v:.4g}" for n, v in zip(NAMES, x[i]))
          + f"  P/기준 {c[i, 0] + 1:.3f}")
    (run_dir / "history.json").write_text(json.dumps(
        {"p_max": p_max, "seconds": sec, "history": history},
        ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
