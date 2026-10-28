"""책 05-1: 학습 데이터의 입력·출력 관계를 살펴본다.

사용법: uv run python ch05/explore.py [출력 폴더]   (기본 ~/cfd-runs/ch05)
결과:   표 출력과 출력 폴더의 explore.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import skew

from data import INPUTS, TARGETS, load

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/ch05")
out = out.expanduser()
out.mkdir(parents=True, exist_ok=True)

print("split   학습 가능 행 수")
for s in ["train", "val", "test", "ood_U", "ood_H"]:
    print(f"{s:<7} {len(load(s)[0]):4d}")

x, y, df = load("train")
print("\n출력      최솟값    최댓값   최대/최소  왜도  로그 왜도")
for i, t in enumerate(TARGETS):
    v = y[:, i]
    print(f"{t:<7} {v.min():9.4f} {v.max():9.4f} {v.max() / v.min():8.1f}"
          f"  {skew(v):5.2f}  {skew(np.log(v)):6.2f}")

print("\n순위 상관계수 (train)")
corr = df[INPUTS + TARGETS].corr(method="spearman").loc[INPUTS, TARGETS]
print(corr.round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
ax.hist(y[:, 0], bins=30, color="#2a78d6")
ax.set(xlabel="pressure drop dp [m2/s2]", ylabel="number of cases",
       title="(a) dp is skewed")
inset = ax.inset_axes([0.5, 0.45, 0.45, 0.45])
inset.hist(np.log(y[:, 0]), bins=30, color="#eb6834")
inset.set_title("log(dp)", fontsize=9)
inset.tick_params(labelsize=8)
for ax, i, name in [(axes[1], 0, "dp [m2/s2]"), (axes[2], 1, "dT_max [K]")]:
    sc = ax.scatter(x[:, 3], y[:, i], c=x[:, 1], s=12, cmap="viridis")
    ax.set(xlabel="inlet velocity U [m/s]", ylabel=name)
axes[1].set(xscale="log", yscale="log", title="(b) dp grows with U")
axes[1].set_xticks([0.05, 0.07, 0.1, 0.15], ["0.05", "0.07", "0.1", "0.15"])
axes[1].minorticks_off()
axes[2].set(title="(c) peak temperature falls with U")
fig.colorbar(sc, ax=axes[1:], label="rib height ratio e/H")
for ax in axes:
    ax.grid(alpha=0.3)
fig.savefig(out / "explore.png", dpi=150, bbox_inches="tight")
