"""책 12-1: 설정 파일 하나로 샘플링부터 CFD 재검증까지 한 번에 실행한다.

단계: samples -> cfd -> dataset -> train -> evaluate -> optimize -> verify
      -> report
- 단계마다 결과를 작업 폴더에 남기고, 실행 기록을 state.json에 적는다.
- 단계의 설정과 앞 단계 결과가 그대로면 건너뛴다. 설정 하나를 바꾸면
  그 설정을 쓰는 단계부터 다시 실행한다.
- 품질 기준(gate)을 넘지 못하면 그 단계에서 멈춘다.
- 출력은 화면과 함께 logs/<단계>.log, 단계 시작·끝은 pipeline.log에 남는다.

사용법: uv run python ch12/pipeline.py [설정 파일] [--from 단계] [--until 단계]
    설정 기본값 ch12/config.toml. OpenFOAM 환경을 불러온 bash에서 실행한다.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import system
from system import al, problem

HERE = Path(__file__).resolve().parent
CH04 = system.ROOT / "ch04"
STAGES = ["samples", "cfd", "dataset", "train", "evaluate", "optimize",
          "verify", "report"]
USES = {"samples": ["samples"], "cfd": ["cfd"], "dataset": [],
        "train": ["train"], "evaluate": ["gate"], "optimize": ["optimize"],
        "verify": ["verify"], "report": []}       # 단계마다 쓰는 설정 표


class GateError(Exception):
    """품질 기준을 넘지 못해 파이프라인을 멈춘다."""


class Tee:
    """print 출력을 화면과 파일에 함께 쓴다."""

    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)
            f.flush()               # 오래 걸리는 단계도 로그를 바로 볼 수 있게

    def flush(self):
        for f in self.files:
            f.flush()


def sh(cmd):
    """하위 프로그램을 실행하며 출력을 한 줄씩 print로 넘긴다."""
    p = subprocess.Popen([sys.executable, *map(str, cmd)], text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end="")
    if p.wait() != 0:
        raise RuntimeError(f"{cmd[0]} 실패 (종료 코드 {p.returncode})")


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() \
        else None


# ---- 단계 -----------------------------------------------------------------
# 단계 함수는 설정(cfg)을 받아 결과 요약 dict를 돌려준다.

def samples(cfg):
    """04-2의 make_samples로 LHS 샘플 목록을 만든다."""
    import make_samples as ms
    c = cfg["samples"]
    ms.SEED, ms.N_IN = c["seed"], c["n"]
    if not c["ood"]:
        ms.OOD = {}
    ms.main(cfg["out"] / "samples.csv")
    return {"rows": len(pd.read_csv(cfg["out"] / "samples.csv"))}


def cfd(cfg):
    """04-3의 run_batch로 해석하고, 04-4의 rerun_slow로 늦은 케이스를 다시."""
    c, cases = cfg["cfd"], cfg["out"] / "cases"
    sh([CH04 / "run_batch.py", cases, "--workers", c["workers"],
        "--samples", cfg["out"] / "samples.csv"])
    if c["rerun_slow"]:
        sh([CH04 / "rerun_slow.py", cases])
    df = pd.read_csv(cases / "results.csv")
    n = df["status"].value_counts()
    return {"ok": int(n.get("ok", 0)), "nc": int(n.get("not_converged", 0)),
            "failed": int(n.get("failed", 0)),       # nc: 미수렴
            "core_h": round(float(df["seconds"].sum()) / 3600, 2)}


def dataset(cfg):
    """04-5의 build_dataset으로 데이터셋을 만들고 재검증 결과(feedback)를 더한다."""
    import build_dataset
    from check_cases import judge
    out = cfg["out"] / "dataset"
    build_dataset.main(cfg["out"] / "cases", out)
    fb = cfg["out"] / "feedback.csv"
    if fb.exists():
        df = pd.read_parquet(out / "cases.parquet")
        new = pd.read_csv(fb)
        verdict = [judge(r) for r in new.astype(str).to_dict("records")]
        new["valid"] = [v for v, _ in verdict]
        new["reason"] = [w for _, w in verdict]
        new["rerun"] = False
        new["dp_pa"] = (new["dp"] * 998).round(3)
        df = pd.concat([df, new[df.columns]], ignore_index=True)
        df.to_parquet(out / "cases.parquet", index=False)
        print(f"재검증 결과 {len(new)}행을 split=feedback으로 더함")
    df = pd.read_parquet(out / "cases.parquet")
    return {"rows": len(df), "valid": int(df["valid"].sum()),
            "feedback": int((df["split"] == "feedback").sum())}


def train(cfg):
    """train과 feedback 행으로 앙상블 MLP를 학습한다 (05-4, 11-1)."""
    c = cfg["train"]
    df = pd.read_parquet(cfg["out"] / "dataset" / "cases.parquet")
    df = df[df["valid"] & df["split"].isin(["train", "feedback"])]
    x, y = df[system.NAMES].to_numpy(float), df[system.TARGETS].to_numpy(float)
    start = time.perf_counter()
    ens = al.fit_ensemble(x, y, members=c["members"], seed=c["seed"],
                          epochs=c["epochs"])
    sec = time.perf_counter() - start
    d = cfg["out"] / "model"
    d.mkdir(exist_ok=True)
    torch.save(ens.model.state_dict(), d / "model.pt")
    from make_samples import RANGES     # 학습 범위 = 샘플링 범위 (04-2)
    meta = {"model": f"앙상블 MLP {c['members']}개 (SiLU 64x3)",
            "members": c["members"], "epochs": c["epochs"],
            "inputs": system.NAMES, "targets": system.TARGETS,
            "input_range": {k: list(RANGES[k]) for k in system.NAMES},
            "x_scaler": {"mean": ens.sx.mean.tolist(),
                         "std": ens.sx.std.tolist()},
            "z_scaler": {"mean": ens.sz.mean.tolist(),
                         "std": ens.sz.std.tolist()},
            "train_rows": len(df), "samples_seed": cfg["samples"]["seed"],
            "dataset_sha256": file_hash(cfg["out"] / "dataset"
                                        / "cases.parquet"),
            "train_seconds": round(sec, 1)}
    (d / "model.json").write_text(json.dumps(meta, ensure_ascii=False,
                                             indent=2) + "\n")
    print(f"train {len(df)}행, 모델 {c['members']}개, 학습 {sec:.1f}초")
    return {"rows": len(df), "seconds": round(sec, 1)}


def evaluate(cfg):
    """test 오차와 보정 배율(val)을 구해 모델에 적고, 품질 기준을 검사한다."""
    g = cfg["gate"]
    df = pd.read_parquet(cfg["out"] / "dataset" / "cases.parquet")
    df = df[df["valid"]]
    sur = system.Surrogate(cfg["out"] / "model")
    sur.k = None                    # 지난 실행의 보정 배율을 쓰지 않는다
    ev = {}
    if sur.members > 1:             # 2kσ 안에 val의 95%가 들도록 (11-1)
        v = df[df["split"] == "val"]
        y, u = sur.predict(v[system.NAMES].to_numpy(float))
        e = np.abs(y / v[system.TARGETS].to_numpy(float) - 1) * 100
        ev["k"] = round(float(np.quantile(e / u, 0.95) / 2), 2)
    print("구분   개수   " + "".join(f"{t:>16}" for t in system.TARGETS))
    for s in ("test", "ood_U", "ood_H"):
        part = df[df["split"] == s]
        if part.empty:
            continue
        y, _ = sur.predict(part[system.NAMES].to_numpy(float))
        e = np.abs(y / part[system.TARGETS].to_numpy(float) - 1) * 100
        ev[s] = {t: [round(float(e[:, i].mean()), 3),
                     round(float(e[:, i].max()), 2)]
                 for i, t in enumerate(system.TARGETS)}
        print(f"{s:<6} {len(part):4d}   " + "".join(
            f"{a:7.3f}% {b:6.2f}%" for a, b in ev[s].values()))
    worst = max(m for m, _ in ev["test"].values())
    worst_max = max(m for _, m in ev["test"].values())
    meta = json.loads((cfg["out"] / "model" / "model.json").read_text())
    meta["evaluation"] = ev
    (cfg["out"] / "model" / "model.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if worst > g["test_mean_pct"] or worst_max > g["test_max_pct"]:
        raise GateError(f"test 오차 평균 {worst:.3f}%, 최대 {worst_max:.2f}%"
                        f" (기준 {g['test_mean_pct']}%, {g['test_max_pct']}%)")
    print(f"품질 기준 통과: test 평균 {worst:.3f}% ≤ {g['test_mean_pct']}%,"
          f" 최대 {worst_max:.2f}% ≤ {g['test_max_pct']}%")
    return {"k": ev.get("k"), "test_worst": round(worst, 3)}


def optimize(cfg):
    """대리모델로 펌핑 동력 제약 아래 최고 온도가 가장 낮은 설계 (10-3)."""
    c = cfg["optimize"]
    sur = system.Surrogate(cfg["out"] / "model")
    r = system.optimize(sur, c["p_max_ratio"], c["gap_min"])
    (cfg["out"] / "optimum.json").write_text(json.dumps(r, indent=2) + "\n")
    d = r["design"]
    print(f"최적 설계 H {d['H']:.3f} e/H {d['e_ratio']:.3f} N {d['N']}"
          f" U {d['U']:.4f}: 예측 dT_max {r['pred']['dT_max']:.3f}K"
          f" (기준 {r['base']['dT_max']:.3f}K), {r['seconds']:.1f}초")
    return {"dT_max": round(r["pred"]["dT_max"], 3),
            "seconds": round(r["seconds"], 1)}


def verify(cfg):
    """최적 설계를 CFD로 다시 해석해 판정한다 (10-5). fail이면 멈춘다."""
    c = cfg["verify"]
    sur = system.Surrogate(cfg["out"] / "model")
    opt = json.loads((cfg["out"] / "optimum.json").read_text())
    r = system.verify(sur, opt["design"], cfg["out"] / "verify",
                      cells=c["cells"], tol=c["tol_pct"],
                      p_max_ratio=cfg["optimize"]["p_max_ratio"],
                      gap_min=cfg["optimize"]["gap_min"],
                      workers=c["workers"])
    (cfg["out"] / "verify.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2) + "\n")
    print(system.verify_report(r, sur.meta["model"]))
    if r["verdict"] == "fail":
        raise GateError("최적 설계가 CFD에서 수렴하지 않음")
    if r["verdict"] == "incomplete":
        raise GateError(next(n for n in r["notes"]
                             if n.startswith("재검증 미완료")))
    return {"verdict": r["verdict"], "seconds": r["seconds"]}


def report(cfg):
    """단계별 결과와 시간을 report.md 하나로 정리한다."""
    state = json.loads((cfg["out"] / "state.json").read_text())
    v = json.loads((cfg["out"] / "verify.json").read_text())
    lines = [f"## 파이프라인 보고서: {cfg['name']}", "",
             "| 단계 | 시간 | 결과 |", "|---|---|---|"]
    for s in STAGES[:-1]:
        st = state[s]
        res = ", ".join(f"{k} {val}" for k, val in st["summary"].items())
        lines.append(f"| {s} | {st['seconds']:.0f}초 | {res} |")
    lines += ["", system.verify_report(v, cfg["name"])]
    (cfg["out"] / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"보고서: {cfg['out'] / 'report.md'}")
    return {}


# ---- 실행 관리 --------------------------------------------------------------

def stage_key(cfg, stage, prev_key):
    """단계의 설정, 앞 단계의 키, 추가 입력으로 만든 지문."""
    parts = {"stage": stage, "prev": prev_key,
             "cfg": {k: cfg[k] for k in USES[stage]}}
    if stage == "dataset":      # 재검증 결과가 쌓이면 데이터셋부터 다시 만든다
        parts["feedback"] = file_hash(cfg["out"] / "feedback.csv")
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=HERE / "config.toml")
    ap.add_argument("--from", dest="start", choices=STAGES,
                    help="이 단계부터 무조건 다시 실행")
    ap.add_argument("--until", choices=STAGES, help="이 단계까지만 실행")
    args = ap.parse_args()
    cfg = system.load_config(args.config)
    out = cfg["out"]
    (out / "logs").mkdir(parents=True, exist_ok=True)
    state_file = out / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    plog = open(out / "pipeline.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        plog.write(line + "\n")
        plog.flush()

    log(f"설정 {args.config} -> {out}")
    key, forced, total = "", False, time.time()
    for stage in STAGES:
        key = stage_key(cfg, stage, key)
        forced = forced or stage == args.start
        if not forced and state.get(stage, {}).get("key") == key:
            log(f"{stage:<9} 건너뜀 (설정과 입력이 같음)")
        else:
            log(f"{stage:<9} 시작")
            start = time.time()
            with open(out / "logs" / f"{stage}.log", "w") as f, \
                    redirect_stdout(Tee(sys.stdout, f)):
                try:
                    summary = globals()[stage](cfg)
                except GateError as err:
                    state.pop(stage, None)
                    state_file.write_text(json.dumps(state, indent=2))
                    log(f"{stage:<9} 멈춤: {err}")
                    sys.exit(2)
            sec = time.time() - start
            state[stage] = {"key": key, "seconds": round(sec, 1),
                            "summary": summary,
                            "finished": f"{datetime.now():%Y-%m-%d %H:%M}"}
            state_file.write_text(json.dumps(state, ensure_ascii=False,
                                             indent=2))
            log((f"{stage:<9} 끝 {sec:7.1f}초  " + " ".join(
                f"{k}={v}" for k, v in summary.items())).rstrip())
        if stage == args.until:
            break
    log(f"전체 {time.time() - total:.0f}초")


if __name__ == "__main__":
    main()
