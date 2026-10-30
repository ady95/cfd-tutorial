"""책 12장 공통: 앞 장에서 만든 조각을 설계 자동화 시스템의 부품으로 묶는다.

- 설정 파일(config.toml)과 작업 폴더 경로
- 대리모델 불러오기: 파이프라인이 만든 앙상블(model.json) 또는 5장 MLP(mlp.json)
- 예측과 함께 신뢰 범위 경고, 추정 오차를 돌려주는 describe() (06-3, 11-1)
- 대리모델 최적화 (10-3)와 CFD 재검증·판정 (10-5)
- 재검증 결과를 다음 학습 데이터로 쌓는 feedback.csv, 보고서
"""
import csv
import importlib.util
import json
import sys
import threading
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for d in ("ch03", "ch04", "ch05", "ch06", "ch10"):
    sys.path.insert(0, str(ROOT / d))
import problem  # noqa: E402  (ch10, ch06 common의 load_mlp를 함께 불러온다)
from data import TARGETS, Scaler  # noqa: E402
from extract import RHO  # noqa: E402

# ch06과 ch11에 같은 이름(common.py)이 있어서 11장 것은 다른 이름으로 불러온다
_spec = importlib.util.spec_from_file_location("al_common",
                                               ROOT / "ch11" / "common.py")
al = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(al)

NAMES = ["H", "e_ratio", "N", "U"]
ERR_WARN = 0.5      # 추정 오차가 이 값 [%]을 넘으면 CFD 확인을 권한다
FEEDBACK_COLS = ["id", "split", "case", "H", "e", "N", "U", "e_ratio",
                 "status", "error", "cells", "iterations", "dp", "dT_max",
                 "dT_avg", "dT_out", "dT_out_energy", "seconds"]
_lock = threading.Lock()        # problem 모듈의 전역 값을 바꾸는 동안 잠근다


def load_config(path):
    """설정 파일을 읽고 결과 폴더(out)를 붙여 돌려준다."""
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    cfg["out"] = Path(cfg["run_dir"]).expanduser() / cfg["name"]
    return cfg


def default_model_dir():
    """파이프라인 결과가 있으면 그 모델을, 없으면 5장 MLP를 쓴다."""
    cfg = load_config(Path(__file__).parent / "config.toml")
    made = cfg["out"] / "model"
    if (made / "model.json").exists():
        return made
    return ROOT / "models" / "surrogate_v1"


class Surrogate:
    """대리모델 하나. predict(x) -> (예측 (n, 3), 추정 오차 % (n, 3) 또는 None)."""

    def __init__(self, model_dir):
        d = Path(model_dir).expanduser()
        self.dir = d
        if (d / "mlp.json").exists():           # 5장 MLP 하나 (불확실성 없음)
            self.meta = json.loads((d / "mlp.json").read_text())
            self.meta["members"] = 1
            self._mlp = problem.load_mlp(d)
            self.k = None
        else:                                   # 파이프라인이 만든 앙상블
            self.meta = json.loads((d / "model.json").read_text())
            model = al.EnsembleMLP(self.meta["members"])
            model.load_state_dict(torch.load(d / "model.pt"))
            sx, sz = Scaler(), Scaler()
            for s, key in ((sx, "x_scaler"), (sz, "z_scaler")):
                s.mean, s.std = (np.array(self.meta[key][k])
                                 for k in ("mean", "std"))
            self._ens = al.Ensemble(model, sx, sz)
            self.k = self.meta.get("evaluation", {}).get("k")
        self.lo, self.hi = (np.array([self.meta["input_range"][n][i]
                                      for n in NAMES]) for i in (0, 1))

    @property
    def members(self):
        return self.meta["members"]

    def predict(self, x, chunk=100_000):
        """한 번에 chunk개씩 계산해 메모리를 아낀다 (최적화 격자는 약 1000만 점)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        ys, us = [], []
        for i in range(0, len(x), chunk):
            part = x[i:i + chunk]
            if self.members == 1:
                ys.append(self._mlp(part))
            else:
                y, u = self._ens.predict(part)
                ys.append(y)
                us.append(u * (self.k or 1))    # 보정 배율을 곱한 2σ의 절반
        return np.vstack(ys), (np.vstack(us) if us else None)

    def warnings(self, x):
        """설계마다 경고 문장 목록: 학습 범위 밖 변수, 틈 제약 (06-3, 10-1)."""
        x = np.atleast_2d(x)
        out = [[] for _ in x]
        for i, j in zip(*np.nonzero((x < self.lo) | (x > self.hi))):
            out[i].append(f"{NAMES[j]}={x[i, j]:g}: 학습 범위 "
                          f"{self.lo[j]:g}~{self.hi[j]:g} 밖")
        for i in np.flatnonzero(problem.gap(x) < problem.GAP_MIN):
            out[i].append(f"틈 {problem.gap(x[i])[0]:.2f}mm: "
                          f"최소 {problem.GAP_MIN:g}mm 미만")
        return out

    def describe(self, x):
        """설계 (n, 4) -> 예측, 펌핑 동력, 틈, 추정 오차, 경고를 담은 dict 목록."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y, err = self.predict(x)
        warns = self.warnings(x)
        rows = []
        for i, v in enumerate(x):
            r = {"design": dict(zip(NAMES, [float(v[0]), float(v[1]),
                                            int(v[2]), float(v[3])])),
                 "dp_pa": round(float(y[i, 0] * RHO), 4),
                 "dT_max": round(float(y[i, 1]), 4),
                 "dT_avg": round(float(y[i, 2]), 4),
                 "P": round(float(problem.pumping_power(y[i, 0], v[0],
                                                        v[3])), 4),
                 "gap": round(float(problem.gap(v)[0]), 4), "error_pct": None,
                 "warnings": warns[i]}
            if err is not None:
                r["error_pct"] = dict(zip(TARGETS, err[i].round(3).tolist()))
                if err[i].max() > ERR_WARN:
                    r["warnings"].append(f"추정 오차 {err[i].max():.2f}%: "
                                         "CFD 확인 권장")
            r["status"] = "warning" if r["warnings"] else "ok"
            rows.append(r)
        return rows


def optimize(sur, p_max_ratio=1.0, gap_min=1.0, steps=(101, 81, 101)):
    """P ≤ 기준 설계 x p_max_ratio에서 dT_max 최소 설계 (10-3의 best_design)."""
    with _lock:
        problem._mlp = lambda x: sur.predict(x)[0]
        problem.GAP_MIN = gap_min
        start = time.perf_counter()
        x = problem.grid(steps)
        base = problem.base_values()
        design, pred = problem.best_design(base["P"] * p_max_ratio, x,
                                           problem.surrogate(x))
        sec = time.perf_counter() - start
    return {"design": dict(zip(NAMES, [float(design[0]), float(design[1]),
                                       int(design[2]), float(design[3])])),
            "pred": pred, "base": base, "p_max": base["P"] * p_max_ratio,
            "seconds": sec}


def verify(sur, design, run_dir, cells=(30, 60), tol=1.0,
           p_max_ratio=None, workers=2):
    """기준 설계와 design을 CFD로 해석하고 예측과 비교해 판정한다 (10-5).

    판정: 'fail' CFD가 수렴하지 않음, 'mismatch' 예측과 CFD(셀 30) 차이가
    tol [%]를 넘음, 'caution' 촘촘한 격자에서 동력 상한을 넘음, 'pass' 그 밖.
    """
    run_dir = Path(run_dir).expanduser()
    x = np.array([problem.BASE, [design[n] for n in NAMES]], dtype=float)
    pred = sur.describe(x)
    start = time.time()
    res = {c: problem.cfd(x, run_dir / f"nh{c}", cells_h=c, workers=workers)
           for c in cells}
    out = {"design": design, "pred": pred[1], "base_pred": pred[0],
           "cfd": {str(c): r[1] for c, r in res.items()},
           "base_cfd": {str(c): r[0] for c, r in res.items()},
           "seconds": round(time.time() - start, 1), "notes": []}
    c0 = res[cells[0]][1]
    if c0["status"] != "ok":
        out["verdict"] = "fail"
        out["notes"].append(f"셀 {cells[0]} CFD {c0['status']}: "
                            "이 설계는 학습 데이터로 쓰지 않는다")
        return out
    errs = {k: (pred[1][k] / c0[k] - 1) * 100 for k in ("dT_max", "P")}
    out["pred_error_pct"] = errs
    out["verdict"] = "pass"
    if max(abs(e) for e in errs.values()) > tol:
        out["verdict"] = "mismatch"
        out["notes"].append(f"예측과 CFD 차이가 {tol:g}%를 넘음: "
                            "데이터를 더해 다시 학습")
    fine = res[cells[-1]][1]
    if p_max_ratio and fine["status"] == "ok" and len(cells) > 1:
        base_fine = res[cells[-1]][0]
        if fine["P"] > base_fine["P"] * p_max_ratio:
            over = (fine["P"] / base_fine["P"] - 1) * 100
            out["notes"].append(f"셀 {cells[-1]}에서 펌핑 동력이 기준보다 "
                                f"{over:+.1f}%: 제약에 격자 오차만큼 여유 필요")
            if out["verdict"] == "pass":
                out["verdict"] = "caution"
    return out


def add_feedback(path, result, run_dir, cells=30):
    """재검증한 CFD(데이터셋과 같은 셀 30 격자) 결과를 feedback.csv에 쌓는다.

    수렴하지 않은 설계는 학습 데이터에 넣지 않고 avoid.csv에 따로 적는다.
    같은 케이스가 이미 있으면 다시 쓰지 않는다. 더한 행 수를 돌려준다.
    """
    path = Path(path)
    have = set()
    if path.exists():
        with open(path, newline="") as f:
            have = {r["case"] for r in csv.DictReader(f)}
    added = 0
    for r in (result["base_cfd"][str(cells)], result["cfd"][str(cells)]):
        rec = json.loads((Path(run_dir).expanduser() / f"nh{cells}"
                          / r["case"] / "result.json").read_text())
        if rec["status"] != "ok":
            avoid = path.with_name("avoid.csv")
            new = not avoid.exists()
            with open(avoid, "a", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                if new:
                    w.writerow(["case", "status"])
                w.writerow([rec["case"], rec["status"]])
            continue
        if rec["case"] in have:
            continue
        rec.update(id=100000 + len(have), split="feedback",
                   e_ratio=round(rec["e"] / rec["H"], 4))
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, FEEDBACK_COLS, extrasaction="ignore",
                               lineterminator="\n")
            if new:
                w.writeheader()
            w.writerow(rec)
        have.add(rec["case"])
        added += 1
    return added


def verify_report(result, model_name):
    """재검증 결과 dict -> 마크다운 보고서 문자열."""
    d, cells = result["design"], list(result["cfd"])
    lines = ["## 설계 재검증 보고서", "",
             f"- 모델: {model_name}",
             f"- 설계: H {d['H']:.3f}mm, e/H {d['e_ratio']:.3f}, "
             f"N {d['N']}, U {d['U']:.4f}m/s",
             f"- 판정: **{result['verdict']}**",
             f"- CFD 시간: {result['seconds']:.0f}초", ""]
    lines += [f"- {n}" for n in result["notes"]]
    lines += ["", "| 값 | 예측 | 추정 오차 | " +
              " | ".join(f"CFD 셀 {c}" for c in cells) + " |",
              "|---|---|---|" + "---|" * len(cells)]
    for k, label in (("dT_max", "dT_max [K]"), ("P", "P [mW/m]")):
        e = result["pred"]["error_pct"]
        # P는 dp에 비례하므로 dp의 추정 오차를 그대로 쓴다
        est = f"{e['dp' if k == 'P' else k]:.2f}%" if e else "-"
        vals = [result["cfd"][c][k] for c in cells]
        lines.append(f"| {label} | {result['pred'][k]:.3f} | {est} | " +
                     " | ".join("-" if v is None else f"{v:.3f}"
                                for v in vals) + " |")
    base = [result["base_cfd"][c] for c in cells]
    if all(b["status"] == "ok" for b in base) and \
            all(result["cfd"][c]["status"] == "ok" for c in cells):
        lines += ["", "| 기준 대비 | 예측 | " +
                  " | ".join(f"CFD 셀 {c}" for c in cells) + " |",
                  "|---|---|" + "---|" * len(cells)]
        dT = [result["pred"]["dT_max"] - result["base_pred"]["dT_max"]] + \
             [result["cfd"][c]["dT_max"] - b["dT_max"]
              for c, b in zip(cells, base)]
        dP = [result["pred"]["P"] / result["base_pred"]["P"] - 1] + \
             [result["cfd"][c]["P"] / b["P"] - 1 for c, b in zip(cells, base)]
        lines.append("| dT_max [K] | " +
                     " | ".join(f"{v:+.3f}" for v in dT) + " |")
        lines.append("| P | " +
                     " | ".join(f"{v * 100:+.1f}%" for v in dP) + " |")
    return "\n".join(lines) + "\n"
