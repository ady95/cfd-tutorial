"""책 04-1: 격자·무작위·LHS 샘플링을 비교한다.

1) 2차원 16점으로 세 방법의 점 배치를 그린다.
2) 이 책의 설계 변수와 같은 4차원에서 약 600점을 뽑아, 변수 하나만 봤을 때
   몇 개의 서로 다른 값을 시험하는지, 빈 구간이 얼마나 되는지,
   전체 공간을 얼마나 고르게 덮는지(불일치도)를 비교한다.

사용법: uv run python ch04/sampling_compare.py [출력 폴더]  (기본 ~/cfd-runs/sampling)
결과:   표 출력과 출력 폴더의 sampling-compare.png
"""
import itertools
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

COLOR = "#2a78d6"


def grid(levels, d):
    """각 변수를 levels 칸으로 나눈 칸 가운데 점을 모든 조합으로 놓는다."""
    x = (np.arange(levels) + 0.5) / levels
    return np.array(list(itertools.product(x, repeat=d)))


def stats(x):
    """변수별 서로 다른 값 수, 변수별 빈 구간 비율, 최소 거리, 불일치도."""
    n, d = x.shape
    bins = np.minimum((x * n).astype(int), n - 1)   # 변수마다 n칸으로 나눔
    empty = np.mean([1 - len(np.unique(bins[:, j])) / n for j in range(d)])
    levels = np.mean([len(np.unique(x[:, j])) for j in range(d)])
    return levels, empty, pdist(x).min(), qmc.discrepancy(x)


out = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/sampling")
out = out.expanduser()
out.mkdir(parents=True, exist_ok=True)

# 1) 2차원 16점 그림
pts = {"Grid (4 x 4)": grid(4, 2),
       "Random": np.random.default_rng(0).random((16, 2)),
       "Latin hypercube": qmc.LatinHypercube(d=2, seed=0).random(16)}
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
for ax, (name, x) in zip(axes, pts.items()):
    for t in np.linspace(0, 1, 17):                 # 변수마다 16칸 구간
        ax.axvline(t, color="0.9", lw=0.8, zorder=0)
        ax.axhline(t, color="0.9", lw=0.8, zorder=0)
    ax.scatter(x[:, 0], x[:, 1], s=60, color=COLOR, zorder=3)
    # 각 변수 하나만 봤을 때 어떤 값을 시험했는지 축 바깥에 눈금으로 표시
    ax.plot(x[:, 0], np.full(16, -0.06), "|", color=COLOR, ms=12,
            clip_on=False)
    ax.plot(np.full(16, -0.06), x[:, 1], "_", color=COLOR, ms=12,
            clip_on=False)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title=name,
           xlabel="variable 1", ylabel="variable 2", xticks=[], yticks=[])
    ax.xaxis.labelpad = ax.yaxis.labelpad = 30
fig.tight_layout()
fig.savefig(out / "sampling-compare.png", dpi=150)

# 2) 4차원 비교. 격자는 변수마다 5단계(5^4 = 625점), 나머지는 600점
print("방법        점 수  변수별 값 수  빈 구간 비율  최소 거리  불일치도")
rows = [("격자", [grid(5, 4)])]
rows.append(("무작위", [np.random.default_rng(s).random((600, 4))
                        for s in range(20)]))
rows.append(("LHS", [qmc.LatinHypercube(d=4, seed=s).random(600)
                     for s in range(20)]))
for name, xs in rows:
    lv, em, md, dc = np.mean([stats(x) for x in xs], axis=0)
    pad = " " * (10 - sum(2 if ord(c) > 0x1100 else 1 for c in name))
    print(f"{name}{pad}{len(xs[0]):5d}  {lv:12.0f}  {em:11.1%}"
          f"  {md:9.3f}  {dc:8.5f}")
print("(무작위와 LHS는 난수 시드 20개의 평균)")
