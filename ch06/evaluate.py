"""책 06-1: 최종 MLP를 test로 평가하고, 지표마다 무엇을 말해 주는지 비교한다.

1) 모델 파일에 적힌 데이터셋 지문이 지금 데이터셋과 같은지 확인한다.
2) 최종 MLP의 val과 test 지표(MAE, RMSE, R², 상대 오차)를 나란히 출력한다.
3) 압력손실이 작은 케이스와 큰 케이스로 나눠 절대 오차와 상대 오차를 비교한다.
4) 여러 모델의 R²와 상대 오차를 비교한다.
5) test 오차를 03-5의 격자 오차와 비교한다.

사용법: uv run python ch06/evaluate.py
"""
import hashlib
import json

import numpy as np

from common import (MODEL_DIR, ROOT, TARGETS, fit_mean, fit_poly, fit_xgb,
                    load, load_mlp, metrics)

RHO = 998.0                     # 물의 밀도 [kg/m3]: dp를 Pa로 바꿀 때
UNIT = {"dp": "Pa", "dT_max": "K", "dT_avg": "K"}
GRID = {"dp": 1.14, "dT_max": 1.50, "dT_avg": 0.23}    # 03-5 격자 오차 [%]

# 1) 데이터셋 지문
meta = json.loads((MODEL_DIR / "mlp.json").read_text())
data = ROOT / "data" / "cooling_v1" / "cases.parquet"
same = hashlib.sha256(data.read_bytes()).hexdigest() == meta["dataset_sha256"]
print(f"모델이 학습한 데이터셋 {meta['dataset']}: 지문 "
      + ("일치" if same else "불일치 - 결과를 믿지 말 것"))

x_tr, y_tr, _ = load("train")
x_va, y_va, _ = load("val")
x_te, y_te, _ = load("test")
mlp = load_mlp()


def in_units(m):
    """dp의 MAE·RMSE를 Pa로 바꾼다 (dp는 압력을 밀도로 나눈 값)."""
    for k in ("MAE", "RMSE"):
        m["dp"][k] *= RHO
    return m


# 2) 최종 MLP의 val과 test
print("\n최종 MLP     MAE         RMSE        R²          평균 상대  최대 상대")
for split, x, y in [("val", x_va, y_va), ("test", x_te, y_te)]:
    m = in_units(metrics(y, mlp(x)))
    for t in TARGETS:
        v = m[t]
        print(f"{split:<5}{t:<7} {v['MAE']:.4f} {UNIT[t]:<2}  "
              f"{v['RMSE']:.4f} {UNIT[t]:<2}  {v['R2']:.6f}    "
              f"{v['rel_mean']:.3f}%    {v['rel_max']:.2f}%")

# 3) 압력손실 크기별: 작은 3분의 1과 큰 3분의 1
p = mlp(x_te)[:, 0]
dp = y_te[:, 0]
small, large = dp < np.quantile(dp, 1 / 3), dp > np.quantile(dp, 2 / 3)
print("\ntest 압력손실   범위 [Pa]      MAE [Pa]   평균 상대 오차")
for name, k in [("작은 1/3", small), ("큰 1/3", large)]:
    ae = np.abs(p[k] - dp[k]) * RHO
    print(f"{name:<9}  {dp[k].min() * RHO:5.1f}~{dp[k].max() * RHO:5.1f}"
          f"     {ae.mean():.4f}     {(ae / (dp[k] * RHO)).mean() * 100:.3f}%")

# 4) 모델별 R²와 상대 오차 (test, 최고 온도 상승)
print("\ntest dT_max   R²          RMSE [K]   평균 상대 오차")
for name, f in [("평균값", fit_mean(x_tr, y_tr)),
                ("3차 다항", fit_poly(x_tr, y_tr, 3)),
                ("XGBoost", fit_xgb(x_tr, y_tr)), ("MLP", mlp)]:
    v = metrics(y_te, f(x_te))["dT_max"]
    pad = 13 - sum(2 if ord(c) > 0x1100 else 1 for c in name)
    print(name + " " * pad + f" {v['R2']:9.6f}   {v['RMSE']:.4f}     "
          f"{v['rel_mean']:6.3f}%")

# 5) 격자 오차와 비교
m = metrics(y_te, mlp(x_te))
print("\n출력     격자 오차(03-5)  test 평균  test 최대  최대가 격자 오차보다")
for t in TARGETS:
    ok = "작음" if m[t]["rel_max"] < GRID[t] else "큼"
    print(f"{t:<8} {GRID[t]:10.2f}%     {m[t]['rel_mean']:6.3f}%   "
          f"{m[t]['rel_max']:6.2f}%   {ok}")
