"""책 07-2: 실제 물리장을 푸리에 모드 몇 개로 나타낼 수 있는지 본다.

FNO의 푸리에 층은 낮은 주파수 모드 몇 개만 남기고 나머지를 버린다.
같은 방법으로 CFD 물리장을 잘라 되돌렸을 때 얼마나 달라지는지 계산한다.
(FNO는 이 층 말고도 비선형 함수와 건너뛰기 연결이 있어 잘린 모드 밖의
모양도 만들 수 있다. 이 표는 푸리에 층 하나의 감을 잡는 용도다.)

사용법: uv run python ch07/spectrum.py [출력 폴더]   (기본 ~/cfd-runs/ch07)
결과:   모드 수별 복원 오차 표와 출력 폴더의 spectrum.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

from fno_data import load

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODES = [(8, 16), (16, 32), (16, 64), (24, 128), (48, 256)]   # (y, x)


def truncate(f, ky, kx):
    """y 방향 ky개, x 방향 kx개 모드만 남기고 되돌린다 (neuraloperator 방식)."""
    F = np.fft.rfft2(f)
    keep = np.zeros_like(F)
    h = ky // 2
    keep[:h, :kx // 2 + 1] = F[:h, :kx // 2 + 1]           # 0 이상 주파수
    keep[-h:, :kx // 2 + 1] = F[-h:, :kx // 2 + 1]         # 음의 주파수
    return np.fft.irfft2(keep, s=f.shape)


def rel_l2(a, b, m):
    return np.sqrt((((a - b) * m) ** 2).sum() / ((b * m) ** 2).sum()) * 100


data = load()
k = np.flatnonzero(data["split"] == "train")[:50]
print(f"train 50개 케이스의 평균 상대 L2 오차 [%] (유체 격자점만)")
print("모드 (y x)   남는 계수 비율      Ux      Uy       p      dT")
ny, nx = data["mask"].shape[1:]
for ky, kx in MODES:
    err = np.array([[rel_l2(truncate(data["fields"][i, c], ky, kx),
                            data["fields"][i, c], data["mask"][i])
                     for c in range(4)] for i in k]).mean(0)
    frac = ky * (kx // 2 + 1) / (ny * (nx // 2 + 1)) * 100
    print(f"{ky:3d} x {kx:3d}   {frac:10.2f}%   "
          + " ".join(f"{e:7.2f}" for e in err))

# 그림: 케이스 하나의 dT와 Ux를 원본, 16 x 64, 48 x 256 모드로
i = k[0]
m = data["mask"][i].astype(bool)
x0, x1 = np.searchsorted(np.linspace(0, 80, nx), [40, 72])
out = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/ch07")
out = out.expanduser()
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 2, figsize=(14, 6.2), sharex=True, sharey=True)
for col, (c, name, cmap) in enumerate([(3, "dT [K]", "inferno"),
                                       (0, "Ux [m/s]", "viridis")]):
    f = data["fields"][i, c]
    for row, (title, g) in enumerate([("CFD", f),
                                      ("16 x 64 modes", truncate(f, 16, 64)),
                                      ("48 x 256 modes",
                                       truncate(f, 48, 256))]):
        ax = axes[row, col]
        show = np.where(m, g, np.nan)[:, x0:x1]
        im = ax.imshow(show, origin="lower", cmap=cmap, aspect="auto",
                       extent=(40, 72, 0, 3), vmin=np.nanmin(f[m]),
                       vmax=f[m].max())
        ax.set_title(f"{name}: {title}", fontsize=10)
        if row == 2:
            ax.set_xlabel("x [mm]")
        if col == 0:
            ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[:, col], shrink=0.8)
fig.suptitle(f"case {data['case'][i]}", fontsize=10)
fig.savefig(out / "spectrum.png", dpi=130)
print("저장:", out / "spectrum.png")
