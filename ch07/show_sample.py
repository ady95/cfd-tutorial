"""책 07-3: 격자 배열로 옮긴 케이스 하나의 입력 2채널과 출력 4채널을 그린다.

사용법: uv run python ch07/show_sample.py [케이스 이름] [출력 폴더]
    기본값  train의 첫 케이스   ~/cfd-runs/ch07
결과:   출력 폴더의 grid-sample.png (x 0~40 mm 구간)
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

from fno_data import load, make_inputs

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

data = load()
name = sys.argv[1] if len(sys.argv) > 1 else \
    data["case"][data["split"] == "train"][0]
out = Path(sys.argv[2] if len(sys.argv) > 2 else "~/cfd-runs/ch07")
out = out.expanduser()
out.mkdir(parents=True, exist_ok=True)
i = int(np.flatnonzero(data["case"] == name)[0])
mask = data["mask"][i]
x_in = make_inputs(mask[None], data["inputs"][i:i + 1, 3])[0]
ny, nx = mask.shape
x1 = np.searchsorted(np.linspace(0, 80, nx), 40)

panels = [("input: mask", x_in[0], "gray"), ("input: U / 0.1 m/s", x_in[1],
                                             "gray")]
panels += [(f"output: {n}", data["fields"][i, c], cmap) for c, (n, cmap) in
           enumerate([("Ux [m/s]", "viridis"), ("Uy [m/s]", "RdBu_r"),
                      ("p [m2/s2]", "cividis"), ("dT [K]", "inferno")])]
fig, axes = plt.subplots(3, 2, figsize=(14, 6.4), sharex=True, sharey=True)
for ax, (title, f, cmap) in zip(axes.ravel(), panels):
    show = f if title.startswith("input") else np.where(mask > 0, f, np.nan)
    kw = {}
    if cmap == "RdBu_r":
        lim = np.nanmax(np.abs(show))
        kw = {"vmin": -lim, "vmax": lim}
    im = ax.imshow(show[:, :x1], origin="lower", cmap=cmap, aspect="auto",
                   extent=(0, 40, 0, 3), **kw)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
for ax in axes[-1]:
    ax.set_xlabel("x [mm]")
for ax in axes[:, 0]:
    ax.set_ylabel("y [mm]")
fig.suptitle(f"{name}: grid {ny} x {nx} (x 0-40 mm shown)", fontsize=10)
fig.savefig(out / "grid-sample.png", dpi=130)
print(f"{name}: 유체 격자점 {int(mask.sum())}개 / 전체 {mask.size}개")
print("저장:", out / "grid-sample.png")
