"""책 10-4: dT_max와 펌핑 동력 P를 함께 줄이는 다목적 최적화.

1) 정답 파레토 프런트: 격자 991만 점에서 지배되지 않는 점, P 수준마다 SLSQP로 다듬음
2) Optuna NSGA-II와 TPE(다목적)로 찾은 프런트가 정답에 얼마나 가까운지
   (기준 설계 P의 1/4~4배 사이 P 수준 40개에서
    'P 이하로 찾은 가장 낮은 dT_max - 정답'의 평균)
3) 후보: A 기준 설계와 같은 P, B 기준 설계와 같은 dT_max, C P 두 배
4) A에서 dT_max가 0.1K(격자 오차 수준) 안에 드는 설계의 범위와,
   그 가운데 학습 범위 끝에서 가장 먼 설계 D

사용법: uv run python ch10/pareto.py [--seeds 3]
결과:   표 출력, ~/cfd-runs/ch10/pareto.png, ~/cfd-runs/ch10/candidates.json
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import optuna

from problem import (BASE, BOUNDS, GAP_MIN, NAMES, base_values, best_design,
                     from_unit, gap, grid, outlet_rise, surrogate)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = Path("~/cfd-runs/ch10").expanduser()


def front_of(f, p):
    """(dT_max, P) 점들 가운데 지배되지 않는 점의 위치 (P 오름차순)."""
    order = np.argsort(p, kind="stable")
    keep, best = [], np.inf
    for i in order:
        if f[i] < best:
            keep.append(i)
            best = f[i]
    return np.array(keep)


def best_under(f, p, levels):
    """P 수준마다 'P ≤ 수준'인 점의 최소 dT_max (levels 길이)."""
    order = np.argsort(p)
    run = np.minimum.accumulate(f[order])
    k = np.searchsorted(p[order], levels, side="right") - 1
    return np.where(k >= 0, run[np.clip(k, 0, None)], np.inf)


def run_mo(sampler, budget):
    """Optuna 다목적 최적화 -> 평가한 모든 점의 (dT_max, P)."""
    def objective(trial):
        u = [trial.suggest_float("H", 0, 1), trial.suggest_float("e", 0, 1),
             (trial.suggest_int("N", 4, 15) - 4) / 11,
             trial.suggest_float("U", 0, 1)]
        s = surrogate(from_unit(np.array(u)))
        return float(s["dT_max"][0]), float(s["P"][0])

    study = optuna.create_study(directions=["minimize", "minimize"],
                                sampler=sampler)
    study.optimize(objective, n_trials=budget)
    v = np.array([t.values for t in study.trials])
    return v[:, 0], v[:, 1]


def min_power(dT_target, x, s, lo, hi):
    """dT_max ≤ dT_target를 만족하는 가장 작은 P의 설계 (이분법)."""
    for _ in range(20):
        mid = (lo + hi) / 2
        if best_design(mid, x, s)[1]["dT_max"] <= dT_target:
            hi = mid
        else:
            lo = mid
    return best_design(hi, x, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    b = base_values()

    start = time.time()
    x = grid()
    s = surrogate(x)
    ok = gap(x) >= GAP_MIN
    xo, fo, po = x[ok], s["dT_max"][ok], s["P"][ok]
    fr = front_of(fo, po)
    edge = (xo[fr, 2] == 15) & np.isclose(
        xo[fr, 1], np.minimum(0.4, 1 - GAP_MIN / xo[fr, 0]), atol=0.003)
    print(f"격자 프런트 {len(fr)}점 ({time.time() - start:.1f}초)"
          f"  P {po[fr].min():.2f}~{po[fr].max():.2f} mW/m\n"
          f"  그중 리브 비율 위 끝이면서 N 15인 점 {edge.mean() * 100:.0f}%")

    levels = np.geomspace(b["P"] / 4, b["P"] * 4, 40)
    ref = np.array([best_design(p, x, s)[1]["dT_max"] for p in levels])
    print(f"정답 프런트 (P 수준 40개, SLSQP 다듬기) {time.time() - start:.0f}초")

    print(f"\n정답과의 평균 차이 [K] (시드 {args.seeds}개 중앙값)"
          "와 그런 설계를 못 찾은 P 수준 수")
    print("방법                평가 200회       1000회   시간(1000회)")
    found = {}
    for name, make in [("Optuna NSGA-II", lambda sd: optuna.samplers
                        .NSGAIISampler(population_size=50, seed=sd)),
                       ("Optuna TPE", lambda sd: optuna.samplers
                        .TPESampler(seed=sd, multivariate=True))]:
        gaps, miss, sec = {200: [], 1000: []}, {200: [], 1000: []}, []
        for sd in range(args.seeds):
            t = time.time()
            f, p = run_mo(make(sd), 1000)
            sec.append(time.time() - t)
            for n in gaps:
                g = best_under(f[:n], p[:n], levels) - ref
                gaps[n].append(g[np.isfinite(g)].mean())
                miss[n].append(int(np.isinf(g).sum()))
            if sd == 0:
                found[name] = (f, p)
        cells = "".join(f"{np.median(gaps[n]):9.3f} / {max(miss[n]):2d}"
                        for n in gaps)
        print(f"{name:<16}{cells}   {np.mean(sec):6.1f}초")

    cands = {"A": best_design(b["P"], x, s),
             "B": min_power(b["dT_max"], x, s, po[fr].min(), b["P"]),
             "C": best_design(2 * b["P"], x, s)}

    fA = cands["A"][1]["dT_max"]
    near = np.flatnonzero(ok & (s["P"] <= b["P"]) & (s["dT_max"] <= fA + 0.1))
    print(f"\nA에서 0.1K 안 (P ≤ 기준): 격자 {len(near):,}점")
    for i, n in enumerate(NAMES):
        print(f"  {n:<8} {x[near, i].min():6.4g} ~ {x[near, i].max():6.4g}")
    # 학습 범위 끝까지의 거리(0~1로 바꾼 값)가 가장 큰 설계. 같으면 dT_max 순
    u = (x[near] - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    margin = np.minimum(u, 1 - u).min(1).round(3)
    j = near[np.lexsort((s["dT_max"][near], -margin))[0]]
    cands["D"] = (x[j], {k: float(a[j]) for k, a in s.items()})
    print(f"  D는 범위 끝까지 거리 {margin.max():.3f} (변수 범위를 0~1로 볼 때)")

    print("\n후보       H  e_ratio   N      U   dT_max  dT_avg      P  출구 상승")
    rows = [("기준", BASE, b)] + [(k, v, sv) for k, (v, sv) in cands.items()]
    for name, v, sv in rows:
        pad = " " * (4 - len(name) - sum(ord(ch) > 127 for ch in name))
        print(f"{name}{pad}   {v[0]:5.3f}  {v[1]:6.4f}  {v[2]:2.0f}  {v[3]:.4f}"
              f"   {sv['dT_max']:6.3f}  {sv['dT_avg']:6.3f}  {sv['P']:5.2f}"
              f"     {outlet_rise(v[0], v[3]):5.3f}")
    (OUT / "candidates.json").write_text(json.dumps(
        {k: dict(zip(NAMES, map(float, v))) | {"pred": sv}
         for k, (v, sv) in cands.items()}, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    f, p = found["Optuna NSGA-II"]
    ax.scatter(p, f, s=6, color="#2a78d6", alpha=0.35, lw=0,
               label="NSGA-II evaluations (1000)")
    ax.plot(po[fr], fo[fr], color="k", lw=1.5, label="Pareto front (grid)")
    ax.scatter(po[fr][edge], fo[fr][edge], s=4, color="#1a9e5f", zorder=3,
               label="front on range edge (e_ratio max, N 15)")
    ax.scatter([b["P"]], [b["dT_max"]], s=60, color="#7a7a7a", zorder=3,
               edgecolor="white", label="baseline")
    offset = {"A": (8, -12), "D": (-16, 4)}
    for k, (v, sv) in cands.items():
        ax.scatter([sv["P"]], [sv["dT_max"]], s=60, color="#eb6834",
                   edgecolor="white", zorder=4)
        ax.annotate(k, (sv["P"], sv["dT_max"]), textcoords="offset points",
                    xytext=offset.get(k, (6, 4)), fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("pumping power P [mW/m]")
    ax.set_ylabel("max wall temperature rise dT_max [K]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pareto.png", dpi=150)
    print(f"\n저장: {OUT / 'pareto.png'}")


if __name__ == "__main__":
    main()
