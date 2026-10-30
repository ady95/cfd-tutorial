"""책 10장 공통: 냉각 채널 설계 최적화 문제를 코드로 정의한다.

- 설계 변수: H, e_ratio, N(정수), U. 범위는 최종 MLP의 학습 범위(신뢰 범위)
- 목적: 가열면 최고 온도 상승 dT_max [K]와 펌핑 동력 P [mW/m]를 줄인다
  P = Δp x 유량 = ρ·dp x U·H (채널 깊이 1m당, 10-2)
- 제약: 리브 끝과 가열면 사이 틈 H - e ≥ 1.0mm (10-1)
  단일 목적 문제에서는 P ≤ 기준 설계의 P를 더한다 (10-3)
- 평가: 대리모델(MLP, 여러 점을 한 번에)과 CFD(ch03/run_case.py) 두 가지
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ch03"))
sys.path.insert(0, str(ROOT / "ch06"))
from common import load_mlp  # noqa: E402
from extract import CP, Q_WALL, RHO  # noqa: E402

NAMES = ["H", "e_ratio", "N", "U"]
BOUNDS = np.array([[1.5, 2.5], [0.2, 0.4], [4, 15], [0.05, 0.15]])
BASE = np.array([2.0, 0.3, 8, 0.1])     # 기준 설계 (03-1의 기준 형상)
GAP_MIN = 1.0       # 리브 끝과 가열면 사이 최소 틈 [mm]
L_HEAT = 0.06       # 가열 길이 [m]

_mlp = None


def pumping_power(dp, H, U):
    """운동학적 압력손실 dp [m²/s²]와 H [mm], U [m/s] -> 펌핑 동력 [mW/m]."""
    return RHO * dp * U * H     # H의 mm->m(1e-3)와 W->mW(1e3)가 상쇄된다


def outlet_rise(H, U):
    """에너지 보존으로 정해지는 출구 평균 온도 상승 [K] (03-3)."""
    return Q_WALL * L_HEAT / (RHO * CP * U * H * 1e-3)


def gap(x):
    """설계 변수 배열 (n, 4) -> 리브 끝과 가열면 사이 틈 [mm]."""
    x = np.atleast_2d(x)
    return x[:, 0] * (1 - x[:, 1])


def surrogate(x):
    """설계 변수 (n, 4) -> dp, dT_max, dT_avg, P를 담은 dict (각 (n,))."""
    global _mlp
    if _mlp is None:
        _mlp = load_mlp()
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = _mlp(x)
    return {"dp": y[:, 0], "dT_max": y[:, 1], "dT_avg": y[:, 2],
            "P": pumping_power(y[:, 0], x[:, 0], x[:, 3])}


def base_values():
    """기준 설계의 대리모델 예측 (스칼라 dict)."""
    return {k: float(v[0]) for k, v in surrogate(BASE).items()}


def grid(steps=(101, 81, 101)):
    """H, e_ratio, U는 steps개 등간격, N은 4~15 정수인 격자 (n, 4)."""
    axes = [np.linspace(*BOUNDS[0], steps[0]),
            np.linspace(*BOUNDS[1], steps[1]),
            np.arange(4, 16), np.linspace(*BOUNDS[3], steps[2])]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 4)


def best_design(p_max, x, s):
    """P ≤ p_max와 틈 제약에서 dT_max가 가장 작은 설계 -> (설계, 예측 dict).

    x, s는 grid()와 그 surrogate 결과. N마다 격자 최적점에서 시작해
    나머지 세 변수를 SLSQP로 다듬고(격자 사이 값), 가장 좋은 N을 고른다.
    """
    from scipy.optimize import minimize
    ok = (s["P"] <= p_max) & (gap(x) >= GAP_MIN)
    best = None
    for N in range(4, 16):
        m = np.flatnonzero(ok & (x[:, 2] == N))
        if len(m) == 0:
            continue
        i = m[np.argmin(s["dT_max"][m])]

        def full(v, N=N):
            return np.array([v[0], v[1], N, v[2]])

        cons = [{"type": "ineq",
                 "fun": lambda v: 1 - surrogate(full(v))["P"][0] / p_max},
                {"type": "ineq", "fun": lambda v: gap(full(v))[0] - GAP_MIN}]
        r = minimize(lambda v: surrogate(full(v))["dT_max"][0],
                     x[i, [0, 1, 3]], method="SLSQP", constraints=cons,
                     bounds=BOUNDS[[0, 1, 3]], options={"eps": 1e-4})
        v = full(r.x) if r.success else x[i]
        sv = surrogate(v)
        if sv["P"][0] > p_max * (1 + 1e-6) or gap(v)[0] < GAP_MIN - 1e-9:
            v, sv = x[i], surrogate(x[i])       # 다듬기가 제약을 어기면 버린다
        if best is None or sv["dT_max"][0] < best[1]["dT_max"][0]:
            best = (v, sv)
    return best[0], {k: float(a[0]) for k, a in best[1].items()}


def from_unit(u):
    """0~1 범위 변수 (n, 4) -> 틈 제약을 늘 만족하는 설계 (n, 4).

    둘째 변수는 e_ratio가 아니라 '허용되는 e_ratio 범위 안의 위치'다.
    H가 정해지면 틈 제약 H(1 - e_ratio) ≥ 1.0 때문에 e_ratio의 위 끝이
    min(0.4, 1 - 1.0/H)로 줄어든다. N은 반올림해 정수로 만든다.
    """
    u = np.atleast_2d(u)
    H = BOUNDS[0, 0] + u[:, 0] * (BOUNDS[0, 1] - BOUNDS[0, 0])
    top = np.minimum(BOUNDS[1, 1], 1 - GAP_MIN / H)
    er = BOUNDS[1, 0] + u[:, 1] * (top - BOUNDS[1, 0])
    N = np.round(BOUNDS[2, 0] + u[:, 2] * (BOUNDS[2, 1] - BOUNDS[2, 0]))
    U = BOUNDS[3, 0] + u[:, 3] * (BOUNDS[3, 1] - BOUNDS[3, 0])
    return np.column_stack([H, er, N, U])


def to_unit(x):
    """from_unit의 역변환."""
    x = np.atleast_2d(x)
    top = np.minimum(BOUNDS[1, 1], 1 - GAP_MIN / x[:, 0])
    u = (x - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    u[:, 1] = (x[:, 1] - BOUNDS[1, 0]) / (top - BOUNDS[1, 0])
    return u


def cfd(x, run_dir, cells_h=30, workers=6):
    """설계 변수 (n, 4)를 CFD로 해석해 surrogate와 같은 키의 dict 목록으로.

    수렴하지 않거나 실패한 케이스는 값 대신 None을 넣는다.
    """
    from concurrent.futures import ThreadPoolExecutor
    import channel_mesh
    from run_case import run_case
    channel_mesh.CELLS_H = cells_h      # 03-5와 같은 방식으로 격자를 바꾼다
    x = np.atleast_2d(x)

    def one(v):
        H, er, N, U = (round(float(a), 4) for a in v)
        r = run_case(H, round(H * er, 4), int(N), U, run_dir)
        out = {"case": r["case"], "status": r["status"],
               "seconds": r["seconds"], "iterations": r.get("iterations")}
        ok = r["status"] == "ok"
        for k in ("dp", "dT_max", "dT_avg"):
            out[k] = r[k] if ok else None
        out["P"] = pumping_power(r["dp"], H, U) if ok else None
        return out

    # 같은 설계가 여러 번 있으면 한 번만 해석한다 (같은 폴더를 동시에 쓰지 않게)
    uniq, idx = np.unique(np.round(x, 4), axis=0, return_inverse=True)
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(one, uniq))
    return [res[i] for i in idx.ravel()]
