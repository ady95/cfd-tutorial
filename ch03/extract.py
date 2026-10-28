"""책 03-3: 해석 결과에서 대리모델이 배울 값을 뽑는다.

사용법: uv run python ch03/extract.py ~/cfd-runs/rib-base
다른 스크립트에서는 extract(케이스 폴더)가 결과를 dict로 돌려준다.
표준 라이브러리만 쓰므로 python3 로 실행해도 된다.
"""
import re
import sys
from pathlib import Path

RHO, CP = 998.0, 4182.0     # 물의 밀도 [kg/m3], 비열 [J/(kg K)]
Q_WALL = 1e4                # 가열면 열유속 [W/m2] (03-2에서 고정)
T_IN = 293.15               # 입구 온도 [K] (03-2에서 고정)


def read_last(case, func):
    """함수 객체 기록에서 마지막 반복의 (반복 번호, 값, 면 넓이)를 읽는다."""
    dat = next((case / "postProcessing" / func / "0").glob("*.dat"))
    area, row = None, None
    for line in dat.read_text().splitlines():
        if line.startswith("# Area"):
            area = float(line.split(":")[1])
        elif line.strip() and not line.startswith("#"):
            row = line.split()
    return int(float(row[0])), float(row[1]), area


def extract(case):
    case = Path(case).expanduser()
    log = (case / "log.foamRun").read_text()
    it, dp, _ = read_last(case, "pSection")
    _, flow, _ = read_last(case, "flowRate")
    _, t_out, _ = read_last(case, "outletT")
    _, t_max, _ = read_last(case, "wallTmax")
    _, t_avg, a_heat = read_last(case, "wallTavg")

    # 에너지 수지: 가열면으로 들어간 열은 모두 물을 데워 출구로 나간다
    # q x 가열면 넓이 = rho x cp x 유량 x 출구 온도 상승
    dt_energy = Q_WALL * a_heat / (RHO * CP * flow)
    converged = re.search(r"converged in \d+ iterations", log) is not None
    return {
        "converged": converged,
        "iterations": it,
        "dp": dp,                   # 운동학적 압력 [m2/s2]
        "dp_pa": dp * RHO,          # 압력 [Pa]
        "dT_out": t_out - T_IN,
        "dT_out_energy": dt_energy,
        "dT_max": t_max - T_IN,
        "dT_avg": t_avg - T_IN,
    }


if __name__ == "__main__":
    r = extract(sys.argv[1])
    err = (r["dT_out"] / r["dT_out_energy"] - 1) * 100
    print(f"반복 {r['iterations']} ({'수렴' if r['converged'] else '미수렴'})")
    print(f"압력손실 {r['dp']:.5f} m2/s2 = {r['dp_pa']:.1f} Pa")
    print(f"출구 온도 상승 {r['dT_out']:.4f} K"
          f" (에너지 수지 {r['dT_out_energy']:.4f} K, 차이 {err:+.2f}%)")
    print(f"가열면 최고 온도 상승 {r['dT_max']:.3f} K")
    print(f"가열면 평균 온도 상승 {r['dT_avg']:.3f} K")
