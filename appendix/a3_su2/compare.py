"""책 부록 A-3: 같은 격자, 같은 조건에서 OpenFOAM과 SU2를 비교한다.

- 03-3의 범위 끝 설계 다섯 개를 높이 30칸 격자로 두 솔버에서 풀고
- 기준 형상은 60칸 격자로도 풀어 격자를 바꿀 때 두 솔버가 어디로 가는지 본다
비교 값: 입구 평균 압력(운동학적), 가열면 최고·평균 온도 상승, 반복 수, 시간
모두 코어 하나(SU2는 스레드 하나)로 푼다. 동시에 6개씩 실행한다.

사용법: uv run python appendix/a3_su2/compare.py [작업 폴더]
    기본값 ~/cfd-runs/a3/compare
"""
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ch03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import channel_mesh  # noqa: E402
from extract import extract, read_last  # noqa: E402
from su2_case import CFL_DEFAULT, run as run_su2  # noqa: E402

DESIGNS = [(1.5, 0.3, 4, 0.05), (1.5, 0.6, 15, 0.1), (2.0, 0.6, 8, 0.1),
           (2.5, 0.5, 15, 0.15), (2.5, 1.0, 4, 0.05)]


def run_foam(H, e, N, U, case, cells_h=30):
    subprocess.run(["rm", "-rf", str(case)], check=True)
    subprocess.run(["cp", "-r", str(ROOT / "ch03/template"), str(case)],
                   check=True)
    channel_mesh.CELLS_H = cells_h
    text, _ = channel_mesh.block_mesh_dict(H, e, N)
    (case / "system/blockMeshDict").write_text(text)
    subprocess.run(["foamDictionary", "0/U", "-entry",
                    "boundaryField/inlet/value", "-set",
                    f"uniform ({U} 0 0)"], cwd=case, check=True,
                   capture_output=True)
    start = time.perf_counter()
    for cmd in (["blockMesh"], ["foamRun"]):
        with open(case / f"log.{cmd[0]}", "w") as f:
            subprocess.run(cmd, cwd=case, stdout=f, stderr=subprocess.STDOUT,
                           check=True)
    sec = time.perf_counter() - start
    r = extract(case)
    func = next(p.name for p in (case / "postProcessing").iterdir()
                if "inlet" in p.name)
    _, p_in, _ = read_last(case, func)
    return {"seconds": sec, "iters": r["iterations"], "p_inlet": p_in,
            "dT_max": r["dT_max"], "dT_avg": r["dT_avg"]}


def job(args):
    solver, (H, e, N, U), cells_h, work = args
    case = work / f"{solver}_H{H}_e{e}_N{N}_U{U}_c{cells_h}"
    if solver == "OpenFOAM":
        return run_foam(H, e, N, U, case, cells_h)
    return run_su2(H, e, N, U, case, cells_h=cells_h, cfl=CFL_DEFAULT)


if __name__ == "__main__":
    work = Path(sys.argv[1] if len(sys.argv) > 1
                else "~/cfd-runs/a3/compare").expanduser()
    work.mkdir(parents=True, exist_ok=True)
    jobs = [(s, d, 30, work) for d in DESIGNS for s in ("OpenFOAM", "SU2")]
    jobs += [(s, DESIGNS[2], 60, work) for s in ("OpenFOAM", "SU2")]
    with ProcessPoolExecutor(6) as pool:
        res = list(pool.map(job, jobs))
    print("설계 (H e N U)        격자 솔버      반복   시간    입구 압력"
          "   최고 온도  평균 온도")
    for (solver, d, cells_h, _), r in zip(jobs, res):
        name = " ".join(f"{v:g}" for v in d)
        print(f"{name:<20} {cells_h:>4} {solver:<9} {r['iters']:>5} "
              f"{r['seconds']:6.1f}초  {r['p_inlet']:.5f}  {r['dT_max']:8.4f}"
              f"  {r['dT_avg']:8.4f}")
    print("\nSU2 - OpenFOAM 차이 [%] (같은 격자)")
    for i in range(0, len(jobs), 2):
        a, b = res[i], res[i + 1]
        d, cells_h = jobs[i][1], jobs[i][2]
        name = " ".join(f"{v:g}" for v in d)
        diff = [(b[k] / a[k] - 1) * 100 for k in ("p_inlet", "dT_max",
                                                  "dT_avg")]
        print(f"{name:<20} {cells_h:>4}   입구 압력 {diff[0]:+.2f}  최고 "
              f"{diff[1]:+.2f}  평균 {diff[2]:+.2f}")
