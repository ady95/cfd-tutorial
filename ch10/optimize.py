"""책 10-3: 같은 단일 목적 문제를 Optuna와 BoTorch로 풀고 결과와 비용을 비교한다.

문제: 펌핑 동력 P ≤ 기준 설계, 틈 H - e ≥ 1.0mm 에서 dT_max 최소화 (대리모델)
- 정답: 격자 991만 점을 한꺼번에 평가한 뒤 SLSQP로 다듬은 최솟값
- 무작위, Optuna TPE, Optuna GPSampler, BoTorch: 평가 100회씩, 시드 5개
기록: 평가 20·50·100회 때 찾은 가장 좋은 dT_max와 정답의 차이, 걸린 시간

사용법: uv run python ch10/optimize.py [--budget 100] [--seeds 5]
결과:   표 출력, ~/cfd-runs/ch10/optimize.png
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import optuna
import torch

from bo import bo_minimize
from problem import NAMES, base_values, best_design, from_unit, grid, surrogate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = Path("~/cfd-runs/ch10").expanduser()
P_MAX = base_values()["P"]


def evaluate(x):
    """설계 (n, 4) -> dT_max (n,), P 제약 값 (n, 1). 제약은 0 이하면 만족."""
    s = surrogate(x)
    return s["dT_max"], (s["P"] / P_MAX - 1)[:, None]


def best_so_far(f, c):
    """평가 순서대로 '지금까지 찾은 제약 만족 설계의 최소 f' (n,)."""
    f = np.where((c <= 0).all(1), f, np.inf)
    return np.minimum.accumulate(f)


def run_optuna(sampler, budget):
    """Optuna로 budget회 평가한다 -> 평가 순서대로 (f, c).

    BoTorch와 같은 0~1 변수 공간에서 탐색한다(N만 정수로 고른다).
    """
    def objective(trial):
        u = [trial.suggest_float("H", 0, 1), trial.suggest_float("e", 0, 1),
             (trial.suggest_int("N", 4, 15) - 4) / 11,
             trial.suggest_float("U", 0, 1)]
        f, c = evaluate(from_unit(np.array(u)))
        trial.set_constraint("P", c[0, 0])     # 0 이하면 만족
        return float(f[0])

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=budget)
    f = np.array([t.value for t in study.trials])
    c = np.array([[t.constraints["P"]] for t in study.trials])
    return f, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    start = time.time()
    x = grid()
    v, sv = best_design(P_MAX, x, surrogate(x))
    best = sv["dT_max"]
    print(f"정답 ({time.time() - start:.1f}초)  "
          + "  ".join(f"{n} {a:.4g}" for n, a in zip(NAMES, v))
          + f"  dT_max {best:.4f}  P {sv['P']:.3f}")

    # 점 100개 남짓의 GP 계산은 작은 행렬 연산이라 스레드가 많으면 오히려
    # 느리다 (BoTorch 평가 50회: 12스레드 161초, 1스레드 38초)
    torch.set_num_threads(1)
    methods = {
        "무작위": lambda s: run_optuna(
            optuna.samplers.RandomSampler(seed=s), args.budget),
        "Optuna TPE": lambda s: run_optuna(optuna.samplers.TPESampler(
            seed=s, multivariate=True), args.budget),
        "Optuna GP": lambda s: run_optuna(optuna.samplers.GPSampler(
            seed=s, deterministic_objective=True), args.budget),
        "BoTorch": lambda s: bo_minimize(evaluate, args.budget, seed=s),
    }
    marks = [20, 50, args.budget]
    curves = {}
    print(f"\n정답과의 차이 [K] (시드 {args.seeds}개 중앙값, 괄호는 최대)")
    print(f"{'방법':<12}" + "".join(f"{m:>11}회" for m in marks)
          + "     시간")
    for name, run in methods.items():
        runs, sec = [], []
        for s in range(args.seeds):
            t = time.time()
            fs, cs = run(s)[:2]
            sec.append(time.time() - t)
            runs.append(best_so_far(fs, cs) - best)
        runs = np.array(runs)
        curves[name] = runs
        cells = []
        for m in marks:
            v = runs[:, m - 1]
            cells.append(f"{np.median(v):6.3f}({v.max():5.3f})")
        pad = 12 - len(name) - sum(ord(ch) > 127 for ch in name)
        print(name + " " * pad + " ".join(cells)
              + f"  {np.mean(sec):6.1f}초")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = {"무작위": "#7a7a7a", "Optuna TPE": "#2a78d6",
              "Optuna GP": "#eb6834", "BoTorch": "#1a9e5f"}
    labels = {"무작위": "Random", "Optuna TPE": "Optuna TPE",
              "Optuna GP": "Optuna GPSampler", "BoTorch": "BoTorch LogCEI"}
    n = np.arange(1, args.budget + 1)
    for name, runs in curves.items():
        r = np.clip(runs, 1e-4, None)
        r = np.where(np.isfinite(r), r, np.nan)
        ax.plot(n, np.nanmedian(r, 0), color=colors[name], lw=2,
                label=labels[name])
        ax.fill_between(n, np.nanmin(r, 0), np.nanmax(r, 0),
                        color=colors[name], alpha=0.15, lw=0)
    ax.axhline(0.1, color="k", ls=":", lw=1)
    ax.text(args.budget, 0.11, "grid error ~1.5% ", ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best dT_max - optimum [K]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "optimize.png", dpi=150)
    print(f"\n저장: {OUT / 'optimize.png'}")


if __name__ == "__main__":
    main()
