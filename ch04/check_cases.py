"""책 04-4: 배치 해석 결과를 자동으로 판정하고 실패 패턴을 본다.

케이스마다 세 가지 기준(수렴, 에너지 수지, 물리적으로 말이 되는 값)으로
학습에 쓸 수 있는지 판정하고, 수렴하지 않은 케이스의 값이 얼마나 흔들렸는지,
반복 상한을 줄이면 무엇을 얻고 잃는지 계산한다.

사용법: uv run python ch04/check_cases.py [작업 폴더]   (기본 ~/cfd-runs/dataset)
결과:   표 출력과 작업 폴더의 failure-pattern.png
다른 스크립트에서는 judge(행)이 (사용 가능 여부, 이유)를 돌려준다.
"""
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

ENERGY_TOL = 0.001          # 출구 온도 상승과 에너지 수지의 허용 상대 차이
MAX_ITER = 5000             # template의 반복 상한 (controlDict endTime)


def judge(r):
    """results.csv의 한 행을 받아 (학습에 쓸 수 있는지, 이유)를 돌려준다."""
    if r["status"] != "ok":
        return False, r["status"]                   # 미수렴·실패
    err = abs(float(r["dT_out"]) / float(r["dT_out_energy"]) - 1)
    if err > ENERGY_TOL:
        return False, f"에너지 수지 {err:.2%} 어긋남"
    dp, t_max, t_avg = (float(r[k]) for k in ("dp", "dT_max", "dT_avg"))
    if not (dp > 0 and 0 < t_avg < t_max):
        return False, "물리적으로 맞지 않는 값"
    return True, ""


def swing(case, func="wallTmax", last=1000):
    """마지막 last회 동안 기록값이 평균 대비 얼마나 오르내렸는지 [%]."""
    dat = next((case / "postProcessing" / func / "0").glob("*.dat"))
    v = np.loadtxt(dat, comments="#")[-last:, 1]
    if func == "wallTmax":
        v = v - 293.15                              # 온도 상승으로 바꿔 비교
    return (v.max() - v.min()) / v.mean() * 100


def main(run_dir):
    rows = list(csv.DictReader(open(run_dir / "results.csv")))
    verdict = [judge(r) for r in rows]
    ok = [r for r, (good, _) in zip(rows, verdict) if good]
    bad = [(r, why) for r, (good, why) in zip(rows, verdict) if not good]
    print(f"전체 {len(rows)}개: 사용 가능 {len(ok)}개, 제외 {len(bad)}개")

    print("\n제외 케이스             split  e/H    에너지 차이  최고 온도 흔들림"
          "  압력 흔들림")
    for r, why in bad:
        case = run_dir / r["case"]
        err = abs(float(r["dT_out"]) / float(r["dT_out_energy"]) - 1)
        print(f"{r['case']:<25}{r['split']:<6} {float(r['e_ratio']):.3f}"
              f"  {err:9.2%}  {swing(case):13.1f}%"
              f"  {swing(case, 'pSection'):10.1f}%")

    # 반복 상한을 줄이면: 수렴하던 케이스를 잃는 대신 미수렴 케이스 시간을 줄인다
    it = np.array([int(r["iterations"]) for r in rows])
    sec = np.array([float(r["seconds"]) for r in rows])
    good = np.array([r["status"] == "ok" for r in rows])
    print("\n반복 상한  잃는 수렴 케이스  계산 시간 합계")
    for cap in [1000, 1500, 2000, 3000, 5000]:
        lost = int(np.sum(good & (it > cap)))
        # 상한에서 멈추면 반복 수에 비례해 시간이 준다고 본다
        total = np.sum(sec * np.minimum(1, cap / it))
        print(f"{cap:8d}  {lost:14d}개  {total / 60:11.1f}분")

    # 실패 패턴 그림: (a) 리브 높이 비율과 유속, (b) 수렴 케이스의 반복 수
    er = np.array([float(r["e_ratio"]) for r in rows])
    u = np.array([float(r["U"]) for r in rows])
    slow = good & (it > 1000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.scatter(er[good & ~slow], u[good & ~slow], s=12, color="0.75",
                label=f"converged ({np.sum(good & ~slow)})")
    ax1.scatter(er[slow], u[slow], s=40, color="#2a78d6", marker="s",
                label=f"converged after 1000+ iterations ({np.sum(slow)})")
    ax1.scatter(er[~good], u[~good], s=70, color="#eb6834", marker="X",
                label=f"not converged ({np.sum(~good)})")
    ax1.axhline(0.15, ls="--", color="0.5", lw=1,
                label="upper limit of design U")
    ax1.set(xlabel="rib height ratio e/H", ylabel="inlet velocity U [m/s]",
            title="(a) where the solver struggles")
    ax1.legend(loc="upper left", fontsize=9)
    bins = np.logspace(np.log10(50), np.log10(MAX_ITER), 30)
    ax2.hist(it[good], bins=bins, color="#2a78d6")
    ax2.axvline(MAX_ITER, ls="--", color="#eb6834", lw=1.5,
                label=f"iteration limit ({np.sum(~good)} cases stop here)")
    ax2.set(xscale="log", xlabel="iterations to converge",
            ylabel="number of cases", title="(b) converged cases")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "failure-pattern.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/dataset")
         .expanduser())
