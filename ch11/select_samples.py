"""책 11-2: 앙상블의 불확실성으로 다음에 해석할 CFD 케이스를 고른다.

- 무작위: 후보 풀에서 무작위로 (비교군)
- 불확실성 상위: 불확실성이 큰 순서대로 k개
- 불확실성 + 다양성: 하나 고를 때마다 그 근처 후보의 점수를 깎아 가며 k개
  (이미 수렴하지 않은 것으로 확인된 설계 근처도 함께 깎는다, 11-3)
train 50개로 학습한 앙상블이 각 방법으로 25개를 고르면 어디를 고르는지
06-5에서 사람이 고른 구간(모서리, 리브 4~5개, 범위 끝)과 비교한다.

사용법: uv run python ch11/select_samples.py   (CPU 약 1분)
결과 그림: ~/cfd-runs/ch11/select.png
"""
from pathlib import Path

import matplotlib
import numpy as np
import torch

from common import corner, fit_ensemble, load, pad, pool, summary, to_unit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("~/cfd-runs/ch11").expanduser()
LENGTH = 0.15       # 다양성 거리 [0~1 단위]: 이 거리 안의 후보 점수를 깎는다


def pick_random(n_cand, k, rng):
    return rng.choice(n_cand, k, replace=False)


def pick_top(u, k):
    return np.argsort(-u)[:k]


def pick_diverse(u, xu, k, avoid=None, length=LENGTH):
    """점수 = 불확실성 x (1 - exp(-d²/2ℓ²)), d는 이미 고른 점까지 거리.

    xu는 후보의 0~1 좌표, avoid는 피할 점(수렴하지 않은 설계)의 0~1 좌표.
    """
    penalty = np.ones(len(u))

    def push(p):
        d2 = ((xu - p) ** 2).sum(axis=1)
        penalty[:] *= 1 - np.exp(-d2 / (2 * length ** 2))

    for p in (avoid if avoid is not None else []):
        push(p)
    chosen = []
    for _ in range(k):
        i = int(np.argmax(u * penalty))
        chosen.append(i)
        push(xu[i])
    return np.array(chosen)


def describe(x):
    """고른 점들이 06-5의 구간에 얼마나 들어가는지 (%)."""
    xu = to_unit(x)
    edge = ((xu[:, [0, 1, 3]] < 0.1) | (xu[:, [0, 1, 3]] > 0.9)).any(axis=1)
    edge |= np.isin(x[:, 2], [4, 15])
    d = np.sqrt(((xu[:, None] - xu[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return {"corner": corner(x).mean() * 100,
            "few": (x[:, 2] <= 5).mean() * 100,
            "edge": edge.mean() * 100, "nn": d.min(axis=1).mean()}


def main():
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, _ = load("train")
    idx = np.random.default_rng(0).permutation(len(x))[:50]
    cand = pool()
    ens = fit_ensemble(x[idx], y[idx])
    _, u = ens.predict(cand)
    u = summary(u)
    xu = to_unit(cand)
    print(f"후보 풀 {len(cand)}개, train 50개로 학습한 앙상블의 불확실성")
    print(f"  중앙값 {np.median(u):.3f}%, 상위 1% {np.quantile(u, 0.99):.3f}%,"
          f" 최대 {u.max():.3f}%")
    print(f"  모서리 후보 평균 {u[corner(cand)].mean():.3f}%,"
          f" 나머지 {u[~corner(cand)].mean():.3f}%")

    k = 25
    picks = {"무작위": pick_random(len(cand), k, np.random.default_rng(0)),
             "불확실성 상위": pick_top(u, k),
             "불확실성+다양성": pick_diverse(u, xu, k)}
    print(f"\n{k}개를 고른 결과  모서리  리브 4~5개  범위 끝  "
          "최근접 거리  불확실성 평균")
    base = describe(cand)
    print(f"{pad('후보 풀 전체', 16)}{base['corner']:6.0f}% {base['few']:9.0f}%"
          f" {base['edge']:8.0f}%         -   {u.mean():10.3f}%")
    for name, i in picks.items():
        d = describe(cand[i])
        print(f"{pad(name, 16)}{d['corner']:6.0f}% {d['few']:9.0f}%"
              f" {d['edge']:8.0f}% {d['nn']:10.3f}   {u[i].mean():10.3f}%")

    # 윗줄: e/H-U 평면(점선 오른쪽 위가 06-5의 모서리), 아랫줄: N-H 평면
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), sharex="row",
                             sharey="row")
    jit = np.random.default_rng(1).uniform(-0.25, 0.25, len(cand))
    titles = ["random", "top uncertainty", "uncertainty + diversity"]
    for col, (title, i) in enumerate(zip(titles, picks.values())):
        for row, (a, b, ja) in enumerate(((1, 3, 0), (2, 0, 1))):
            ax = axes[row, col]
            xa = cand[:, a] + ja * jit
            ax.scatter(xa, cand[:, b], s=3, c="#d0d0d0", label="candidates")
            ax.scatter(x[idx, a] + ja * jit[:50], x[idx, b], s=18,
                       marker="s", c="#2a6fdb", label="training data (50)")
            ax.scatter(xa[i], cand[i, b], s=42, c="#e8710a", edgecolors="k",
                       linewidths=0.5, label=f"selected ({k})")
            if row == 0:
                ax.axvline(0.35, color="k", ls=":", lw=1)
                ax.axhline(0.12, color="k", ls=":", lw=1)
                ax.set_title(title)
        axes[0, col].set_xlabel("rib height ratio e/H")
        axes[1, col].set_xlabel("number of ribs N")
    axes[0, 0].set_ylabel("inlet velocity U [m/s]")
    axes[1, 0].set_ylabel("channel height H [mm]")
    axes[0, 0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "select.png", dpi=150)
    print(f"\n그림: {OUT / 'select.png'}")


if __name__ == "__main__":
    main()
