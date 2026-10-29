"""책 09-2: CFD 격자 배열과 FNO 예측이 질량·에너지 보존을 얼마나 지키는지 잰다.

단면마다 질량 잔차 Q/Q_in - 1과 에너지 잔차 (E - E_기대)/E_TOTAL을 구하고
(physics.py), 케이스마다 단면 평균 |잔차|를 낸 뒤 split별로 평균한다.
- CFD 격자 배열: 보간과 적분 공식이 만드는 바닥값. 물리 손실이 이보다
  작게 만들라고 요구하면 데이터와 다투게 된다
- FNO: 7장에서 학습한 모델의 예측

사용법: uv run python ch09/conservation.py [FNO 모델]   (기본 ~/cfd-runs/ch07/fno)
결과:   표 출력과 ~/cfd-runs/ch09/conservation.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

from physics import E_TOTAL, all_weights, residuals

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch07"))
from fno_data import load  # noqa: E402
from fno_model import load_fno, predict  # noqa: E402
from grid import GRAD_T, axes  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = ["train", "val", "test", "ood_U", "ood_H"]
D_T = 1.4376e-7                     # 열확산계수 [m²/s] (03-2)
model_path = Path(sys.argv[1] if len(sys.argv) > 1
                  else "~/cfd-runs/ch07/fno").expanduser()
out = Path("~/cfd-runs/ch09").expanduser()
out.mkdir(parents=True, exist_ok=True)

data = load()
model, scaler, _ = load_fno(model_path)

print("단면 평균 |잔차| [%] (케이스 평균), 괄호는 케이스별 단면 최대의 평균")
print("split   개수   대상           질량            에너지")
keep = {}
for s in SPLITS:
    k = data["split"] == s
    W, q, e = all_weights(data, k)
    f = data["fields"][k]
    pred = predict(model, scaler, data["mask"][k], data["inputs"][k, 3])
    for name, g in [("CFD 격자", f), ("FNO (7장)", pred)]:
        rm, re = residuals(g[:, 0], g[:, 3], W, q, e)
        rm, re = np.abs(rm) * 100, np.abs(re) * 100
        print(f"{s:7s} {k.sum():4d}   {name:10s}"
              f"{rm.mean(1).mean():7.2f} ({rm.max(1).mean():5.2f})"
              f"{re.mean(1).mean():9.2f} ({re.max(1).mean():5.2f})")
        if s == "test":
            keep[name] = (rm, re, g)
    if s == "test":
        keep["W"], keep["q"], keep["e"] = W, q, e
        keep["idx"] = np.flatnonzero(k)

# 경계조건과 흐름 방향 열확산 (test)
k = data["split"] == "test"
nx = data["mask"].shape[2]
x, y = axes(nx, data["mask"].shape[1])
dy = (y[1] - y[0]) * 1e-3                          # m
heated = (x >= 10) & (x <= 70)
U = data["inputs"][k, 3]
print("\n경계조건과 흐름 방향 열확산 (test 평균)")
for name, g in [("CFD 격자", data["fields"][k]), ("FNO (7장)", keep["FNO (7장)"][2])]:
    slip = np.abs(g[:, 0, 0, :]).mean(1) / U * 100           # 가열면 줄의 |Ux| / U
    T0, T1, T2 = g[:, 3, 0, heated], g[:, 3, 1, heated], g[:, 3, 2, heated]
    grad = -(-3 * T0 + 4 * T1 - T2) / (2 * dy)               # 벽으로 들어오는 기울기
    gerr = np.abs(grad / GRAD_T - 1).mean(1) * 100
    print(f"  {name:10s} 가열면 |Ux|/U {slip.mean():6.3f}%   "
          f"가열면 온도 기울기 오차 {gerr.mean():6.1f}%")
g = data["fields"][k]
diff = D_T * keep["W"] * np.gradient(g[:, 3], x * 1e-3, axis=2)
ratio = np.abs(diff.sum(1)).max(1) / E_TOTAL * 100
print(f"  흐름 방향 열확산 / 전체 가열량: 단면 최대값의 케이스 평균 "
      f"{ratio.mean():.4f}%")

# 그림: test에서 FNO 질량 잔차가 중앙값인 케이스의 단면별 유량·열 유량
rm_fno = keep["FNO (7장)"][0].mean(1)
j = np.argsort(rm_fno)[len(rm_fno) // 2]
i = keep["idx"][j]
x, _ = axes(data["mask"].shape[2], 2)
W, q, e = keep["W"][j], keep["q"][j], keep["e"][j]
fig, axes_ = plt.subplots(1, 2, figsize=(13, 4))
for name, style in [("CFD 격자", "-"), ("FNO (7장)", "--")]:
    g = keep[name][2][j]
    label = "CFD (grid)" if name == "CFD 격자" else "FNO (ch. 7)"
    axes_[0].plot(x, (W * g[0]).sum(0) / q, style, label=label)
    axes_[1].plot(x, (W * g[0] * g[3]).sum(0) * 1e3, style, label=label)
axes_[0].set(xlabel="x (mm)", ylabel="Q(x) / (U H)",
             title="mass: flow rate through each section")
axes_[1].plot(x, e * 1e3, ":", color="k", label="heat added upstream")
axes_[1].set(xlabel="x (mm)", ylabel="E(x) (K m/s mm) x 1e3",
             title="energy: heat flux through each section")
for ax in axes_:
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle(f"test case {data['case'][i]}", fontsize=10)
fig.tight_layout()
fig.savefig(out / "conservation.png", dpi=130)
print("\n그림 케이스:", data["case"][i])
print("저장:", out / "conservation.png")
