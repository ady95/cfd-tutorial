"""책 11-4, 11-5: al_loop.py 기록(history.json)을 모아 선택 방법을 비교한다.

- 11-4: 50개에서 시작해 300개까지 (무작위, 불확실성 상위, 불확실성+다양성)
- 11-5: 418개 전체에서 100개 더하기 (무작위, 06-5 수동 판단, 불확실성+다양성)
- 11-5: 넓힌 범위의 새 영역에 100개 더하기 (무작위, 불확실성+다양성)
값은 시드 3개의 평균이고, 괄호 안은 가장 나쁜 시드다.

사용법: uv run python ch11/compare.py
결과 그림: ~/cfd-runs/ch11/al_curve.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

from common import pad

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AL = Path("~/cfd-runs/ch11/al").expanduser()
NAMES = {"random": "무작위", "top": "불확실성 상위",
         "diverse": "불확실성+다양성", "manual": "06-5 수동 판단"}
COLORS = {"random": "#7f7f7f", "top": "#e8710a", "diverse": "#2a6fdb",
          "manual": "#1e9e5a"}
LABELS = {"random": "random", "top": "top uncertainty",
          "diverse": "uncertainty + diversity"}


def runs(group, strategy):
    """시드별 history 목록."""
    return [json.loads(p.read_text()) for p in
            sorted((AL / group).glob(f"{strategy}_s*/history.json"))]


def stack(hs, key):
    """(시드, 반복) 배열."""
    return np.array([[r[key] for r in h] for h in hs], dtype=float)


def cfd_minutes(hs):
    """시드별 CFD 케이스 시간 합 [분]과 그중 미수렴 케이스 시간 [분]."""
    total = np.array([h[-1]["cfd_seconds"] for h in hs]) / 60
    fail = np.array([sum(r.get("fail_seconds", 0) for r in h)
                     for h in hs]) / 60
    return total, fail


def picked(hs, key):
    """고른 점 중 구간 비율 [%] (모든 반복·시드 평균)."""
    return np.mean([r["picked"][key] for h in hs for r in h if "picked" in r])


def cum_minutes(r, cap=1.0):
    """반복마다 그때까지의 CFD 시간 합 [분]. cap은 미수렴 케이스 시간 배율."""
    fail = np.cumsum([0] + [x.get("fail_seconds", 0) for x in r[:-1]])
    return (np.array([x["cfd_seconds"] for x in r]) - (1 - cap) * fail) / 60


def at_time(h, key, minutes, cap=1.0):
    """CFD 시간 합이 minutes분일 때의 값 (시드별, 반복 사이는 로그 보간)."""
    out = []
    for r in h:
        t = cum_minutes(r, cap)
        v = np.log([x[key] for x in r])
        out.append(np.exp(np.interp(minutes, t, v)) if t[-1] >= minutes
                   else np.nan)
    return np.array(out)


def loop_part():
    print("== 11-4: 50개에서 시작, 25개씩 10번 (새 시험 200개의 평균 상대 오차)")
    hs = {s: runs("base_n50", s) for s in ("random", "top", "diverse")}
    print(pad("더한 횟수", 16) + "".join(f"{k:>8d}" for k in range(0, 11, 2)))
    for s, h in hs.items():
        m = stack(h, "test_mean")
        print(pad(NAMES[s], 16) + "".join(f"{v:7.3f}%" for v in
                                         m.mean(0)[::2]))

    print("\n10번 더한 뒤    데이터   평균 오차      최대 오차     모서리  범위 밖")
    for s, h in hs.items():
        m, w = stack(h, "test_mean")[:, -1], stack(h, "test_max")[:, -1]
        c, o = stack(h, "corner_mean")[:, -1], stack(h, "ood_mean")[:, -1]
        print(f"{pad(NAMES[s], 16)}{stack(h, 'n')[:, -1].mean():5.0f}개"
              f" {m.mean():6.3f}% ({m.max():.3f}) {w.mean():5.2f}% ({w.max():.2f})"
              f" {c.mean():6.3f}% {o.mean():6.3f}%")

    # 무작위 300개의 평균 오차에 다른 방법이 처음 닿는 데이터 수
    goal = stack(hs["random"], "test_mean")[:, -1].mean()
    print(f"\n무작위 300개의 평균 오차 {goal:.3f}%에 처음 닿는 데이터 수")
    for s, h in hs.items():
        m = stack(h, "test_mean").mean(0)
        k = np.flatnonzero(m <= goal)
        n = stack(h, "n").mean(0)
        print(f"  {pad(NAMES[s], 16)} {f'{n[k[0]]:.0f}개' if len(k) else '-'}")

    # 비교 시점: 무작위 시드 셋이 모두 다다른 CFD 시간 (상한을 낮춘 경우 포함)
    t_goal = min(cum_minutes(r, 0.4)[-1] for r in hs["random"])
    print(f"\nCFD 비용 (케이스 시간 합)과 같은 CFD 시간({t_goal:.1f}분)일 때의 오차")
    print("방법            미수렴  CFD 시간  그중 미수렴   평균 오차  최대 오차")
    for s, h in hs.items():
        tot, fail = cfd_minutes(h)
        print(f"{pad(NAMES[s], 16)}{stack(h, 'failed')[:, -1].mean():4.1f}개"
              f" {tot.mean():7.1f}분 {fail.mean():8.1f}분"
              f" {at_time(h, 'test_mean', t_goal).mean():10.3f}%"
              f" {at_time(h, 'test_max', t_goal).mean():8.2f}%")

    # 04-4: 계산 시간은 반복 수에 비례하므로 상한 2000회면 미수렴 시간은 0.4배
    print(f"\n반복 상한을 5000회에서 2000회로 낮췄다면 (미수렴 시간 x 0.4로 추정)")
    print("방법            CFD 시간  같은 시간의 평균 오차  최대 오차")
    for s, h in hs.items():
        tot = np.mean([cum_minutes(r, 0.4)[-1] for r in h])
        print(f"{pad(NAMES[s], 16)}{tot:7.1f}분"
              f" {at_time(h, 'test_mean', t_goal, 0.4).mean():19.3f}%"
              f" {at_time(h, 'test_max', t_goal, 0.4).mean():10.2f}%")

    print("\n고른 점의 구간 비율 (모든 반복 평균)   모서리  리브 4~5개  범위 끝")
    for s, h in hs.items():
        print(f"  {pad(NAMES[s], 36)} {picked(h, 'corner'):5.0f}%"
              f" {picked(h, 'few'):9.0f}% {picked(h, 'edge'):8.0f}%")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for s, h in hs.items():
        n = stack(h, "n").mean(0)
        t = np.array([[x["cfd_seconds"] for x in r] for r in h]).mean(0) / 60
        for ax, key, xs in zip(axes, ("test_mean", "test_max", "test_mean"),
                               (n, n, t)):
            v = stack(h, key)
            ax.plot(xs, v.mean(0), "-o", ms=3, color=COLORS[s],
                    label=LABELS[s])
            ax.fill_between(xs, v.min(0), v.max(0), color=COLORS[s],
                            alpha=0.15, lw=0)
    h418 = runs("base_n418", "random")      # 11-5 실험의 출발점 (train 전체)
    if h418:
        full = stack(h418, "test_mean")[:, 0].mean()
        for ax in (axes[0], axes[2]):
            ax.axhline(full, color="k", ls="--", lw=1,
                       label=f"all 418 training cases ({full:.3f}%)")
    axes[0].set(ylabel="mean relative error [%]", xlabel="training cases",
                title="mean error vs. data")
    axes[1].set(ylabel="max relative error [%]", xlabel="training cases",
                title="worst error vs. data")
    axes[2].set(ylabel="mean relative error [%]",
                xlabel="added CFD time (sum of case times) [min]",
                title="mean error vs. CFD time")
    for ax in axes:
        ax.set(yscale="log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(AL.parent / "al_curve.png", dpi=150)
    print(f"\n그림: {AL.parent / 'al_curve.png'}")


def add_part():
    print("\n== 11-5: 418개에서 100개 더하기 (새 시험 200개, 오차는 평균 상대 오차)")
    print("방법             전체 평균       전체 최대     모서리 범위 밖"
          "  CFD 시간(미수렴)")
    h0 = runs("base_n418", "random")
    m0, w0 = stack(h0, "test_mean")[:, 0], stack(h0, "test_max")[:, 0]
    print(f"{pad('더하기 전', 16)} {m0.mean():6.3f}% ({m0.max():.3f})"
          f" {w0.mean():5.2f}% ({w0.max():.2f})"
          f" {stack(h0, 'corner_mean')[:, 0].mean():6.3f}%"
          f" {stack(h0, 'ood_mean')[:, 0].mean():6.3f}%")
    for s in ("random", "manual", "diverse"):
        h = runs("base_n418", s)
        m, w = stack(h, "test_mean")[:, -1], stack(h, "test_max")[:, -1]
        tot, _ = cfd_minutes(h)
        print(f"{pad(NAMES[s], 16)} {m.mean():6.3f}% ({m.max():.3f})"
              f" {w.mean():5.2f}% ({w.max():.2f})"
              f" {stack(h, 'corner_mean')[:, -1].mean():6.3f}%"
              f" {stack(h, 'ood_mean')[:, -1].mean():6.3f}%"
              f" {tot.mean():5.1f}분({stack(h, 'failed')[:, -1].mean():.1f}개)")


def wide_part():
    print("\n== 11-5: 넓힌 범위의 새 영역에 100개 더하기 (넓힌 범위 시험 96개)")
    print("방법             새 영역 평균    새 영역 최대    원래 범위"
          "  CFD 시간(미수렴)")
    h0 = runs("wide_n418", "random")
    m0, w0 = stack(h0, "wide_mean")[:, 0], stack(h0, "wide_max")[:, 0]
    print(f"{pad('더하기 전', 16)} {m0.mean():6.3f}% ({m0.max():.3f})"
          f" {w0.mean():5.2f}% ({w0.max():5.2f})"
          f" {stack(h0, 'test_mean')[:, 0].mean():6.3f}%")
    hs = {s: runs("wide_n418", s) for s in ("random", "diverse")}
    for s, h in hs.items():
        m, w = stack(h, "wide_mean")[:, -1], stack(h, "wide_max")[:, -1]
        tot, fail = cfd_minutes(h)
        print(f"{pad(NAMES[s], 16)} {m.mean():6.3f}% ({m.max():.3f})"
              f" {w.mean():5.2f}% ({w.max():5.2f})"
              f" {stack(h, 'test_mean')[:, -1].mean():6.3f}%"
              f"  {tot.mean():5.1f}분({stack(h, 'failed')[:, -1].mean():.1f}개)")

    # 비교 시점: 무작위 시드 셋이 모두 다다른 CFD 시간 (반복 상한별로 따로)
    t5, t2 = (min(cum_minutes(r, cap)[-1] for r in hs["random"])
              for cap in (1.0, 0.4))
    print("\n같은 CFD 시간일 때 새 영역 평균 오차")
    print(f"  반복 상한          5000회({t5:.1f}분)  2000회({t2:.1f}분, 추정)")
    for s, h in hs.items():
        print(f"  {pad(NAMES[s], 16)}"
              f" {at_time(h, 'wide_mean', t5).mean():13.3f}%"
              f" {at_time(h, 'wide_mean', t2, 0.4).mean():17.3f}%")


if __name__ == "__main__":
    loop_part()
    if runs("base_n418", "diverse"):
        add_part()
    if runs("wide_n418", "diverse"):
        wide_part()
