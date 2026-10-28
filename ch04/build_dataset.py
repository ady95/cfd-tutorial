"""책 04-5: 배치 해석 결과로 대리모델 학습용 데이터셋을 만든다.

~/cfd-runs/dataset/results.csv를 읽어 재해석(rerun) 결과로 바꿔 치기하고,
check_cases.judge로 학습 사용 여부를 판정한 뒤, 입력·출력·메타데이터 열을
정리해 Parquet와 CSV로 저장한다. 판정에서 제외된 케이스도 지우지 않고
valid 열로 표시해 남긴다. 만든 환경과 파일 지문은 meta.json에 남긴다.

사용법: uv run python ch04/build_dataset.py [작업 폴더] [출력 폴더]
    기본값  ~/cfd-runs/dataset          data/cooling_v1 (저장소 안)
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from check_cases import judge

VERSION = "cooling_v1"
SPLITS = ["train", "val", "test", "ood_U", "ood_H"]
INPUTS = {"H": "채널 높이 [mm]", "e": "리브 높이 [mm]",
          "e_ratio": "리브 높이 비율 e/H", "N": "리브 개수",
          "U": "입구 유속 [m/s]"}
OUTPUTS = {"dp": "압력손실, x=5mm 단면 평균 운동학적 압력 [m2/s2]",
           "dT_max": "가열면 최고 온도 상승 [K]",
           "dT_avg": "가열면 평균 온도 상승 [K]"}
META = {"id": "샘플 번호 (samples.csv)", "split": "train/val/test/ood_U/ood_H",
        "case": "케이스 폴더 이름", "valid": "학습에 써도 되는지 (04-4 판정)",
        "reason": "valid가 False인 이유", "status": "ok/not_converged/failed",
        "rerun": "반복 상한을 늘려 다시 해석한 결과인지",
        "iterations": "수렴까지 반복 수", "seconds": "케이스 해석 시간 [초]",
        "cells": "격자 셀 수", "dp_pa": "압력손실 [Pa] (dp x 998)",
        "dT_out": "출구 온도 상승 [K] (검증용)",
        "dT_out_energy": "에너지 수지로 계산한 출구 온도 상승 [K]"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run_dir, out):
    df = pd.read_csv(run_dir / "results.csv")
    df["rerun"] = False
    # 04-4에서 다시 해석해 수렴한 케이스는 그 결과로 바꾼다
    for f in sorted((run_dir / "rerun").glob("*/result.json")):
        r = json.loads(f.read_text())
        if r["status"] != "ok":
            continue
        i = df.index[df["case"] == r["case"]][0]
        for k in ["status", "iterations", "seconds", "dp", "dp_pa", "dT_max",
                  "dT_avg", "dT_out", "dT_out_energy"]:
            df.loc[i, k] = r[k]
        df.loc[i, "rerun"] = True
    df["dp_pa"] = (df["dp"] * 998).round(3)
    verdict = [judge(r) for r in df.astype(str).to_dict("records")]
    df["valid"] = [v for v, _ in verdict]
    df["reason"] = [why for _, why in verdict]

    df = df[list(META)[:3] + list(INPUTS) + list(OUTPUTS) + list(META)[3:]]
    df = df.astype({"N": "int64", "iterations": "Int64", "cells": "Int64",
                    "valid": "bool", "rerun": "bool"})
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "cases.parquet", index=False)
    df.to_csv(out / "cases.csv", index=False, lineterminator="\n")

    env = json.loads((run_dir / "env.json").read_text())
    env.pop("date")                 # 날짜가 바뀌어도 같은 지문이 나오게 뺀다
    meta = {
        "version": VERSION,
        "description": "2D 리브 냉각 채널 OpenFOAM 해석 결과 (책 4장)",
        "rows": len(df),
        "valid_rows": {s: int(df.loc[df["split"] == s, "valid"].sum())
                       for s in SPLITS},
        "columns": {"inputs": INPUTS, "outputs": OUTPUTS, "meta": META},
        "fixed_conditions": {"q": "10 kW/m2", "T_in": "20 C",
                             "cells_height": 30},
        "simulation_env": env,
        "sha256": {f: sha256(out / f) for f in ["cases.parquet",
                                                 "cases.csv"]},
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    print(f"{out}: {len(df)}행, 학습 사용 가능 {int(df['valid'].sum())}행")
    for s in SPLITS:
        g = df[df["split"] == s]
        print(f"  {s:<6} {len(g):4d}행  사용 가능 {int(g['valid'].sum()):4d}")
    for f in ["cases.parquet", "cases.csv", "meta.json"]:
        print(f"  {f:<14} {(out / f).stat().st_size / 1024:6.1f} KB")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/dataset")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "data" / VERSION
    main(run.expanduser(), dst)
