"""책 06-3: 학습 범위 밖(OOD) 조건에서 모델별 오차가 어떻게 변하는지 본다.

1) test(범위 안)와 ood_U(U 0.15~0.20), ood_H(H 2.5~3.0)의 상대 오차를 비교한다.
2) 범위 경계에서 멀어질수록 오차가 얼마나 커지는지 구간별로 나눠 본다.
3) 범위 검사와 최근접 거리로 OOD 입력을 얼마나 골라내는지 확인한다.

사용법: uv run python ch06/ood.py [출력 폴더]   (기본 ~/cfd-runs/ch06)
결과:   표 출력과 출력 폴더의 ood.png
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

from common import (MODEL_DIR, Scaler, fit_poly, fit_xgb, load, load_mlp,
                    metrics)

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

x_tr, y_tr, _ = load("train")
models = {"3차 다항": fit_poly(x_tr, y_tr, 3),
          "6차 다항": fit_poly(x_tr, y_tr, 6),
          "XGBoost": fit_xgb(x_tr, y_tr), "MLP": load_mlp()}
data = {s: load(s) for s in ("test", "ood_U", "ood_H")}


def width(s):
    return sum(2 if ord(c) > 0x1100 else 1 for c in s)   # 한글은 2칸


# 1) 범위 안과 밖의 상대 오차
print("평균 상대 오차 [%] dp/dT_max/dT_avg (괄호는 세 출력 중 최대)")
print("모델      " + "".join(f"{s:<23}" for s in data))
for name, f in models.items():
    line = name + " " * (10 - width(name))
    for x, y, _ in data.values():
        m = metrics(y, f(x))
        cell = "/".join(f"{v['rel_mean']:.2f}" for v in m.values())
        cell += f" ({max(v['rel_max'] for v in m.values()):.1f})"
        line += f"{cell:<23}"
    print(line.rstrip())

# 2) 경계에서 멀어질수록: 최고 온도 상승의 평균 상대 오차
for split, col, edges in [("ood_U", 3, [0.15, 0.1625, 0.175, 0.1875, 0.2]),
                          ("ood_H", 0, [2.5, 2.625, 2.75, 2.875, 3.0])]:
    x, y, _ = data[split]
    print(f"\n{split}: 구간별 dT_max 평균 상대 오차 [%]")
    print("구간          개수" + "".join(f"{n:>10}" for n in models))
    preds = {n: f(x) for n, f in models.items()}
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x[:, col] >= lo) & (x[:, col] < hi)
        err = [np.abs(p[k, 1] / y[k, 1] - 1).mean() * 100
               for p in preds.values()]
        print(f"{lo:g}~{hi:g}".ljust(14) + f"{k.sum():4d}"
              + "".join(f"{e:10.2f}" for e in err))

# 3) OOD 감지: 범위 검사와 train까지의 최근접 거리
rng = json.loads((MODEL_DIR / "mlp.json").read_text())["input_range"]
lo, hi = np.array(list(rng.values())).T
sx = Scaler().fit(x_tr)


def nearest(x):
    """표준화한 입력 공간에서 가장 가까운 train 샘플까지의 거리."""
    d = sx.transform(x)[:, None, :] - sx.transform(x_tr)[None, :, :]
    return np.sqrt((d ** 2).sum(axis=2)).min(axis=1)


limit = np.percentile(nearest(data["test"][0]), 95)
print(f"\nOOD 감지 (최근접 거리 기준값 = test의 95번째 백분위 {limit:.3f})")
print("데이터    범위 밖 판정   최근접 거리 초과   최근접 거리 중앙값")
for s, (x, _, _) in data.items():
    out = ((x < lo) | (x > hi)).any(axis=1)
    d = nearest(x)
    print(f"{s:<9} {out.sum():3d}/{len(x):<3d}        {(d > limit).sum():3d}/"
          f"{len(x):<3d}          {np.median(d):.3f}")

# 그림: 경계 밖으로 갈수록 dT_max 상대 오차
out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/ch06")
out_dir = out_dir.expanduser()
out_dir.mkdir(parents=True, exist_ok=True)
colors = {"3차 다항": "#2a78d6", "6차 다항": "#eb6834", "XGBoost": "#7a7a7a",
          "MLP": "#1a9e5f"}
labels = {"3차 다항": "poly 3", "6차 다항": "poly 6", "XGBoost": "XGBoost",
          "MLP": "MLP"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (split, col, name, edge) in zip(axes, [
        ("ood_U", 3, "inlet velocity U [m/s]", 0.15),
        ("ood_H", 0, "channel height H [mm]", 2.5)]):
    x = np.vstack([data["test"][0], data[split][0]])
    y = np.vstack([data["test"][1], data[split][1]])
    for n, f in models.items():
        err = np.abs(f(x)[:, 1] / y[:, 1] - 1) * 100
        ax.scatter(x[:, col], err, s=12, color=colors[n], label=labels[n],
                   alpha=0.8)
    ax.axvspan(edge, x[:, col].max() * 1.01, color="0.9", zorder=0)
    ax.text(edge, 0.0015, " outside training range", fontsize=9)
    ax.set(xlabel=name, yscale="log", ylim=(0.001, 300),
           title=f"({'ab'[col == 0]}) dT_max error, test + {split}")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("relative error [%]")
axes[0].legend(loc="upper left")
fig.tight_layout()
fig.savefig(out_dir / "ood.png", dpi=150)
