"""책 11-5: 데이터를 더해 넓힌 설계 공간에서 다시 최적화하고 CFD로 확인한다.

1. 수렴 지도: 넓힌 범위에서 해석한 모든 케이스(시험 데이터, 능동학습)의
   수렴 여부를 e/H와 U 구간별로 센다.
2. 넓힌 모델: train 418개 + 새 영역에서 수렴한 케이스 전부로 앙상블을 학습하고
   넓힌 범위 시험 데이터로 평가한다.
3. 다시 최적화: 10-3과 같은 문제(펌핑 동력 ≤ 기준, 틈 ≥ 1mm, dT_max 최소)를
   원래 범위와 넓힌 범위에서 풀고, 두 최적 설계를 CFD(셀 30, 60)로 재해석한다.
   수렴하지 않은 설계에서 0~1 거리 0.15 안은 후보에서 뺀다 (11-3과 같은 거리).

사용법: uv run python ch11/expand.py   (OpenFOAM 환경 필요, 약 5분)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from common import (GAP_MIN, RANGES, ROOT, WIDE, fit_ensemble, load, pad,
                    to_unit)
from new_test import load_new_test, load_wide_test
from select_samples import LENGTH

sys.path.insert(0, str(ROOT / "ch03"))
from extract import RHO  # noqa: E402

AL = Path("~/cfd-runs/ch11/al/wide_n418").expanduser()
RUN = Path("~/cfd-runs/ch11/expand").expanduser()
BASE = np.array([2.0, 0.3, 8, 0.1])     # 기준 설계 (10-1)


def wide_data():
    """넓힌 범위 능동학습에서 해석한 케이스 (중복 제거): 수렴 x, y와 미수렴 x."""
    xs, ys, bad = [], [], []
    for f in sorted(AL.glob("*_s*/*/result.json")):     # 케이스 폴더마다 하나
        row = json.loads(f.read_text())
        v = [row["H"], round(row["e"] / row["H"], 4), row["N"], row["U"]]
        if row["status"] == "ok":
            xs.append(v)
            ys.append([row[k] for k in ("dp", "dT_max", "dT_avg")])
        else:
            bad.append(v)
    x, i = np.unique(np.array(xs), axis=0, return_index=True)
    return x, np.array(ys)[i], np.unique(np.array(bad).reshape(-1, 4), axis=0)


def convergence_map(x_ok, x_bad):
    print("수렴 지도 (넓힌 범위의 새 영역, 해석한 케이스 전부)")
    print("  e/H 구간      U < 0.1        U ≥ 0.1      (미수렴/해석 수)")
    x = np.vstack([x_ok, x_bad])
    bad = np.r_[np.zeros(len(x_ok), bool), np.ones(len(x_bad), bool)]
    for lo, hi in ((0.2, 0.4), (0.4, 0.45), (0.45, 0.5)):
        cells = []
        for fast in (False, True):
            m = (x[:, 1] > lo) & (x[:, 1] <= hi) & ((x[:, 3] >= 0.1) == fast)
            cells.append(f"{bad[m].sum():3d}/{m.sum():3d}")
        print(f"  {lo:.2f}~{hi:.2f}      {cells[0]:>8}       {cells[1]:>8}")
    print(f"  (e/H 0.4 이하는 리브 16~20개인 케이스)")


def best_design(ens, ranges, avoid, p_max):
    """격자에서 N마다 최적점을 찾고 SLSQP로 다듬어 가장 좋은 설계."""
    def pred(v):
        m = ens.predict(np.atleast_2d(v))[0]
        return m[:, 1], RHO * m[:, 0] * v[..., 3] * v[..., 0]

    def far(v):
        if len(avoid) == 0:
            return np.full(len(v), np.inf)
        d = ((to_unit(v, WIDE)[:, None] - to_unit(avoid, WIDE)[None]) ** 2)
        return np.sqrt(d.sum(-1)).min(axis=1)

    axes = [np.linspace(*ranges["H"], 51), np.linspace(*ranges["e_ratio"], 51),
            None, np.linspace(*ranges["U"], 51)]
    best = None
    for N in range(ranges["N"][0], ranges["N"][1] + 1):
        axes[2] = [N]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 4)
        g = g[(g[:, 0] * (1 - g[:, 1]) >= GAP_MIN) & (far(g) >= LENGTH)]
        dT, P = pred(g)
        ok = np.flatnonzero(P <= p_max)
        if len(ok) == 0:
            continue
        v0 = g[ok[np.argmin(dT[ok])]]

        def full(v, N=N):
            return np.array([v[0], v[1], N, v[2]])

        cons = [{"type": "ineq", "fun": lambda v: 1 - pred(full(v))[1][0]
                 / p_max},
                {"type": "ineq", "fun": lambda v: full(v)[0]
                 * (1 - full(v)[1]) - GAP_MIN},
                {"type": "ineq", "fun": lambda v: far(full(v)[None])[0]
                 - LENGTH}]
        bounds = [ranges["H"], ranges["e_ratio"], ranges["U"]]
        r = minimize(lambda v: pred(full(v))[0][0], v0[[0, 1, 3]],
                     method="SLSQP", bounds=bounds, constraints=cons,
                     options={"eps": 1e-4})
        v = full(r.x) if r.success else v0
        if pred(v)[1][0] > p_max * (1 + 1e-6):
            v = v0
        if best is None or pred(v)[0][0] < best[1]:
            best = (v, pred(v)[0][0], pred(v)[1][0])
    return best


def cfd(designs, cells_h):
    """설계 목록을 셀 cells_h 격자로 해석 (10-5와 같은 방법)."""
    import channel_mesh
    from run_case import run_case
    channel_mesh.CELLS_H = cells_h

    def one(v):
        H, er, N, U = (round(float(a), 4) for a in v)
        return run_case(H, round(H * er, 4), int(N), U, RUN / f"cells{cells_h}")
    with ThreadPoolExecutor(len(designs)) as pool:
        return list(pool.map(one, designs))


def main():
    torch.set_num_threads(6)
    x_tr, y_tr, _ = load("train")
    x_al, y_al, bad_al = wide_data()
    x_wt, y_wt, bad_wt = load_wide_test()
    x_in, y_in = load_new_test()
    convergence_map(np.vstack([x_al, x_wt]), np.vstack([bad_al, bad_wt]))

    allv = pd.read_parquet(ROOT / "data" / "cooling_v1" / "cases.parquet")
    bad_old = allv.loc[~allv["valid"], ["H", "e_ratio", "N", "U"]].to_numpy()
    avoid = np.vstack([bad_al, bad_wt, bad_old])
    print(f"\n피할 설계(수렴하지 않은 설계) {len(avoid)}개")

    x, y = np.vstack([x_tr, x_al]), np.vstack([y_tr, y_al])
    print(f"넓힌 모델 학습: train {len(x_tr)}개 + 새 영역 {len(x_al)}개")
    ens = fit_ensemble(x, y)
    for name, xs, ys in (("원래 범위 새 시험", x_in, y_in),
                         ("넓힌 범위 시험", x_wt, y_wt)):
        e = np.abs(ens.predict(xs)[0] / ys - 1) * 100
        print(f"  {name:<12} 평균 {e.mean():.3f}%, 최대 {e.max():.2f}%")

    m = ens.predict(BASE[None])[0][0]
    p_max = RHO * m[0] * BASE[3] * BASE[0]
    print(f"\n기준 설계 예측: dT_max {m[1]:.3f} K, P {p_max:.3f} mW/m")
    results = {}
    for name, ranges in (("원래 범위", RANGES), ("넓힌 범위", WIDE)):
        v, dT, P = best_design(ens, ranges, avoid, p_max)
        results[name] = v
        print(f"{name} 최적: H {v[0]:.3f}, e/H {v[1]:.3f}, N {int(v[2])},"
              f" U {v[3]:.4f}")
        print(f"  예측 dT_max {dT:.3f} K (기준 대비 {dT - m[1]:+.3f} K),"
              f" P {P:.3f} mW/m")

    designs = [BASE] + list(results.values())
    names = ["기준"] + list(results)
    print("\nCFD 재해석   격자    상태  dT_max [K]  P [mW/m]  기준 대비")
    for cells in (30, 60):
        rs = cfd(designs, cells)
        base = rs[0]
        for name, v, r in zip(names, designs, rs):
            if r["status"] != "ok":
                print(f"{pad(name, 12)} 셀 {cells}  {r['status']}")
                continue
            P = RHO * r["dp"] * v[3] * v[0]
            d = r["dT_max"] - base["dT_max"]
            print(f"{pad(name, 12)} 셀 {cells}  ok  {r['dT_max']:10.3f}"
                  f" {P:9.3f}  {d:+.3f} K")


if __name__ == "__main__":
    main()
