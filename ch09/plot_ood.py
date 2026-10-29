"""책 09-4: 범위 밖 높이(ood_H) 케이스 하나에서 보존과 정확도를 함께 본다.

7장 FNO와 물리 손실 모델의 온도 상승 dT를 CFD와 나란히 그리고, 단면별 유량
Q(x)/(U H)도 함께 그린다. 케이스는 ood_H에서 7장 FNO의 dT 오차가 중앙값인 것.

사용법: uv run python ch09/plot_ood.py [이름=경로 ...]
    기본값  7장=~/cfd-runs/ch07/fno  λ1+넓은입력=~/cfd-runs/ch09/full/lam1_wide
결과:   ~/cfd-runs/ch09/ood-case.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

from physics import all_weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch07"))
from fno_data import load  # noqa: E402
from fno_model import load_fno, predict, rel_l2  # noqa: E402
from grid import axes  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT = ["7장=~/cfd-runs/ch07/fno",
           "λ1+넓은입력=~/cfd-runs/ch09/full/lam1_wide"]
LABEL = {"7장": "FNO (ch. 7)", "λ1+넓은입력": "FNO + physics loss (wide inputs)"}

data = load()
k = np.flatnonzero(data["split"] == "ood_H")
mask, U = data["mask"][k], data["inputs"][k, 3]
true = data["fields"][k]
preds = {}
for item in sys.argv[1:] or DEFAULT:
    name, path = item.split("=")
    model, scaler, _ = load_fno(Path(path).expanduser())
    preds[name] = predict(model, scaler, mask, U)
first = next(iter(preds))
err = rel_l2(preds[first], true, mask)[:, 3]
j = np.argsort(err)[len(err) // 2]
W, q, _ = all_weights(data, k[j:j + 1])

nx, ny = mask.shape[2], mask.shape[1]
x, _ = axes(nx, ny)
x0, x1 = np.searchsorted(x, [40, 72])
m = mask[j].astype(bool)
rows = [("CFD", true[j])] + [(LABEL.get(n, n), p[j]) for n, p in preds.items()]
fig, axes_ = plt.subplots(len(rows) + 1, 1, figsize=(12, 2.2 * len(rows) + 3))
t = true[j, 3]
for ax, (title, g) in zip(axes_, rows):
    im = ax.imshow(np.where(m, g[3], np.nan)[:, x0:x1], origin="lower",
                   cmap="inferno", aspect="auto", extent=(40, 72, 0, 3),
                   vmin=t[m].min(), vmax=t[m].max())
    e = rel_l2(g[None], true[j:j + 1], mask[j:j + 1])[0, 3]
    ax.set_title(f"{title}: dT (K)" + ("" if title == "CFD"
                                        else f", L2 error {e:.1f}%"),
                 fontsize=10)
    ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, shrink=0.9)
ax = axes_[-1]
for title, g in rows:
    ax.plot(x, (W[0] * g[0]).sum(0) / q[0], label=title)
ax.set(xlabel="x (mm)", ylabel="Q(x) / (U H)", xlim=(0, 80),
       title="flow rate through each section")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.suptitle(f"ood_H case {data['case'][k[j]]}", fontsize=10)
fig.tight_layout()
out = Path("~/cfd-runs/ch09/ood-case.png").expanduser()
fig.savefig(out, dpi=130)
print("케이스:", data["case"][k[j]])
print("저장:", out)
