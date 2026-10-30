"""검수에서 찾은 결함(R01~R07)이 고쳐졌는지 짧게 확인한다.

CFD와 학습은 돌리지 않는다. 해석 결과와 무거운 계산은 가짜 함수로 바꿔 끼우고,
판정·검증·캐시 코드가 그 결과를 어떻게 다루는지만 본다. OpenFOAM 없이 돈다.

사용법: uv run python tests/check_fixes.py
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pydantic
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ch12"))
import system  # noqa: E402
from system import al, problem  # noqa: E402
import run_case as rc  # noqa: E402  (system이 ch03을 경로에 넣었다)
from check_cases import judge  # noqa: E402

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"{'통과' if cond else '실패'}  {name}")


def raises(fn, err=Exception):
    try:
        fn()
    except err:
        return True
    return False


# ---- 가짜 대리모델과 가짜 CFD -------------------------------------------------
class FakeSur:
    def describe(self, x, gap_min=system.GAP_MIN):
        return [{"dT_max": 7.0, "P": 10.0, "error_pct": None} for _ in x]


def ok(P=10.0, dT=7.0):
    return {"case": "c", "status": "ok", "P": P, "dT_max": dT}


def fake_cfd(table):
    """table[셀] = (기준 설계 결과, 후보 결과)를 돌려주는 problem.cfd."""
    return lambda x, run_dir, cells_h, workers: list(table[cells_h])


def verdict(table, **kw):
    problem.cfd = fake_cfd(table)
    design = {"H": 2.0, "e_ratio": 0.3, "N": 8, "U": 0.1}
    return system.verify(FakeSur(), design, tempfile.mkdtemp(), **kw)


# ---- R01: 재검증 CFD 가운데 하나라도 실패하면 pass가 아니다 -------------------
fail = {"case": "c", "status": "failed", "P": None, "dT_max": None}
r = verdict({30: (ok(), ok()), 60: (ok(), fail)}, p_max_ratio=1.0)
check("R01 후보 셀 60 실패 -> incomplete", r["verdict"] == "incomplete")
r = verdict({30: (ok(), ok()), 60: (fail, ok())}, p_max_ratio=1.0)
check("R01 기준 셀 60 실패 -> 예외 없이 incomplete",
      r["verdict"] == "incomplete")
r = verdict({30: (fail, ok()), 60: (ok(), ok())}, p_max_ratio=1.0)
check("R01 기준 셀 30 실패 -> incomplete", r["verdict"] == "incomplete")
r = verdict({30: (ok(), fail), 60: (ok(), ok())}, p_max_ratio=1.0)
check("R01 후보 셀 30 실패 -> fail", r["verdict"] == "fail")
r = verdict({30: (ok(), ok(P=math.nan)), 60: (ok(), ok())}, p_max_ratio=1.0)
check("R01 후보 셀 30 값이 NaN -> fail", r["verdict"] == "fail")
r = verdict({30: (ok(), ok(dT=7.5)), 60: (ok(), fail)}, p_max_ratio=1.0)
check("R01 mismatch와 미완료 사유를 함께 남김",
      r["verdict"] == "mismatch" and "incomplete" in r["reasons"])
r = verdict({30: (ok(), ok()), 60: (ok(), ok())}, p_max_ratio=1.0)
check("R01 모두 수렴, 제약 안 -> pass", r["verdict"] == "pass")

# ---- R02: 요청한 동력 상한·최소 틈으로 판정한다 ------------------------------
r = verdict({30: (ok(), ok()), 60: (ok(10.0), ok(9.0))}, p_max_ratio=0.8)
check("R02 상한 0.8에서 0.9배 -> caution", r["verdict"] == "caution")
r = verdict({30: (ok(), ok()), 60: (ok(10.0), ok(10.5))}, p_max_ratio=2.0)
check("R02 상한 2.0에서 1.05배 -> pass", r["verdict"] == "pass")
r = verdict({30: (ok(), ok()), 60: (ok(), ok())}, p_max_ratio=1.0,
            gap_min=1.5)            # 설계의 틈은 2.0 x (1 - 0.3) = 1.4mm
check("R02 틈 1.4mm < 최소 1.5mm -> caution", r["verdict"] == "caution")
r = verdict({30: (ok(), ok()), 60: (ok(), ok())})
check("R02 제약 없이 부르면 검사하지 않았다고 적음",
      any("검사하지 않음" in n for n in r["notes"]))

# ---- R04: 격자 목록 검사 ---------------------------------------------------
for bad in ([], [30], [60, 30], [-1], [30, 30], [20, 60]):
    check(f"R04 cells={bad} 거부",
          raises(lambda: verdict({30: (ok(), ok()), 60: (ok(), ok())},
                                 cells=bad), ValueError))
check("R04 피드백은 셀 30만",
      raises(lambda: system.add_feedback("x.csv", {}, ".", cells=60),
             ValueError))

# ---- R03: 파이프라인 형식 모델, 멤버 하나 -----------------------------------
rng = np.random.default_rng(0)
lo, hi = np.array([1.5, 0.2, 4, 0.05]), np.array([2.5, 0.4, 15, 0.15])
x = lo + rng.random((40, 4)) * (hi - lo)
y = np.c_[0.04 + x[:, 3], 7 + x[:, 1], 5 + x[:, 1]]
for members in (1, 2):
    ens = al.fit_ensemble(x, y, members=members, epochs=50)
    d = Path(tempfile.mkdtemp())
    torch.save(ens.model.state_dict(), d / "model.pt")
    meta = {"model": "test", "members": members,
            "input_range": {n: [float(a), float(b)]
                            for n, a, b in zip(system.NAMES, lo, hi)},
            "x_scaler": {"mean": ens.sx.mean.tolist(),
                         "std": ens.sx.std.tolist()},
            "z_scaler": {"mean": ens.sz.mean.tolist(),
                         "std": ens.sz.std.tolist()}}
    (d / "model.json").write_text(json.dumps(meta))
    sur = system.Surrogate(d)
    pred, err = sur.predict(x[:3])
    check(f"R03 멤버 {members}개 저장->불러오기->예측",
          pred.shape == (3, 3) and (err is None) == (members == 1))

# ---- R07: 최적화의 최소 틈이 다음 요청에 남지 않는다 -------------------------
seen = {}


def fake_best(p_max, x, s):
    seen["gap"] = problem.GAP_MIN
    return np.array([2.0, 0.3, 8, 0.1]), {"dT_max": 7.0}


problem.grid = lambda steps: np.zeros((1, 4))
problem.base_values = lambda: {"P": 10.0}
problem.surrogate = lambda x: {}
problem.best_design = fake_best
system.optimize(FakeSur(), gap_min=0.5)
check("R07 최적화 안에서는 요청한 틈 0.5mm",
      seen["gap"] == 0.5)
check("R07 최적화 뒤 전역 틈은 1.0mm로 돌아옴", problem.GAP_MIN == 1.0)
warn = system.Surrogate.warnings(sur, np.array([[2.0, 0.6, 8, 0.1]]))
check("R07 예측 경고는 기본 틈 1.0mm 기준",
      any("최소 1mm" in w for w in warn[0]))
problem.best_design = lambda *a: 1 / 0
raises(lambda: system.optimize(FakeSur(), gap_min=0.3))
check("R07 최적화가 예외로 끝나도 전역 틈을 되돌림", problem.GAP_MIN == 1.0)

# ---- R06: 비유한값과 빠진 값 -------------------------------------------------
row = {"status": "ok", "dp": "0.04", "dT_max": "7.6", "dT_avg": "5.3",
       "dT_out": "0.7188", "dT_out_energy": "0.7188"}
check("R06 정상 행은 통과", judge(row)[0])
for k, v in (("dT_out", "nan"), ("dp", "inf"), ("dT_max", "nan"),
             ("dT_out_energy", "0"), ("dT_avg", ""), ("dp", None)):
    check(f"R06 {k}={v!r} 거부", not judge({**row, k: v})[0])
check("R06 열이 빠진 행 거부",
      not judge({k: v for k, v in row.items() if k != "dp"})[0])

# ---- R05: 입력이 바뀌면 저장된 결과를 다시 쓰지 않는다 -----------------------
calls = []


def fake_run(cmd, case, name):
    calls.append(name)
    if name == "checkMesh":
        (case / "log.checkMesh").write_text("Mesh OK.\n")


rc.run = fake_run
rc.extract = lambda case: {"converged": True, "dp": 0.04}
work, tpl = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / "template"
shutil.copytree(rc.TEMPLATE, tpl)
rc.TEMPLATE = tpl
first = rc.run_case(2.0, 0.6, 8, 0.1, work)
calls.clear()
again = rc.run_case(2.0, 0.6, 8, 0.1, work)
check("R05 같은 입력이면 해석하지 않음", not calls and again == first)
cdict = tpl / "system" / "controlDict"
cdict.write_text(cdict.read_text() + "\n// 바꾼 템플릿\n")
calls.clear()
third = rc.run_case(2.0, 0.6, 8, 0.1, work)
check("R05 템플릿이 바뀌면 다시 해석",
      "foamRun" in calls and third["inputs"] != first["inputs"])
check("R05 옛 결과는 _stale로 옮겨 둠",
      (work / "_stale" / f"H2_e0.6_N8_U0.1_{first['inputs']}"
       / "result.json").exists())
saved = work / "H2_e0.6_N8_U0.1" / "result.json"
legacy = {k: v for k, v in third.items() if k != "inputs"}
saved.write_text(json.dumps(legacy))
calls.clear()
check("R05 지문이 없는 옛 결과는 그대로 씀",
      rc.run_case(2.0, 0.6, 8, 0.1, work) == legacy and not calls)

# ---- API 요청 검사 (R02, R04) -----------------------------------------------
try:
    import api
except Exception as err:            # 모델 파일이 없는 환경
    print(f"건너뜀  API 검사 ({type(err).__name__}: {err})")
else:
    d = {"H": 2.0, "e_ratio": 0.3, "N": 8, "U": 0.1}
    for bad in ([], [30], [60, 30], [-1]):
        check(f"API cells={bad} 거부",
              raises(lambda: api.VerifyRequest(design=d, cells=bad),
                     pydantic.ValidationError))
    got = {}
    api.system = type("S", (), {
        "verify": staticmethod(lambda *a, **kw: got.update(kw) or
                               {"verdict": "pass"}),
        "verify_report": staticmethod(lambda *a: "")})
    req = api.VerifyRequest(design=d, p_max_ratio=0.8, gap_min=1.2,
                            feedback=False)
    job = "checkfixes"
    (api.JOBS / f"{job}.json").write_text("{}")
    api.run_job(job, req)
    for f in api.JOBS.glob(f"{job}.*"):
        f.unlink()
    check("API /verify가 요청한 제약을 그대로 넘김",
          got.get("p_max_ratio") == 0.8 and got.get("gap_min") == 1.2)

bad = [n for n, c in results if not c]
print(f"\n{len(results) - len(bad)}/{len(results)} 통과")
sys.exit(1 if bad else 0)
