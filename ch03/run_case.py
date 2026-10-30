"""책 03-4: 설계 변수 하나로 냉각 채널 해석을 처음부터 끝까지 실행한다.

템플릿 복사 -> 변수 치환 -> 격자 생성·검사 -> 해석 -> 결과 추출을 차례로 하고,
케이스 폴더에 로그와 result.json을 남긴다. 어느 단계에서 실패해도 예외를
밖으로 던지지 않고 status와 error에 기록한다.

사용법: uv run python ch03/run_case.py H e N U [작업 폴더]
    예: uv run python ch03/run_case.py 2 0.6 8 0.1 ~/cfd-runs/auto
다른 스크립트에서는 run_case(H, e, N, U, 작업 폴더)가 결과를 dict로 돌려준다.
OpenFOAM 환경을 불러온 bash에서 실행한다 (책 02-1).
"""
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from channel_mesh import block_mesh_dict
from extract import extract

TEMPLATE = Path(__file__).resolve().parent / "template"
TIMEOUT = 600       # 프로그램 하나에 허용하는 최대 시간 [초]


def case_name(H, e, N, U):
    return f"H{H:g}_e{e:g}_N{N}_U{U:g}"


def run(cmd, case, name):
    """프로그램 하나를 케이스 폴더에서 실행하고 출력을 log.<name>에 남긴다."""
    with open(case / f"log.{name}", "w") as log:
        try:
            p = subprocess.run(cmd, cwd=case, stdout=log,
                               stderr=subprocess.STDOUT, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{name} 시간 초과 ({TIMEOUT}초)")
    if p.returncode in (-signal.SIGINT, -signal.SIGTERM):
        raise InterruptedError(f"{name} 중단됨")    # Ctrl+C 등 사용자 중단
    if p.returncode != 0:
        raise RuntimeError(f"{name} 실패 (종료 코드 {p.returncode})")


def inputs_hash(text, U):
    """해석 입력의 지문: 템플릿 파일 전체, blockMeshDict, 입구 유속."""
    h = hashlib.sha256()
    for f in sorted(p for p in TEMPLATE.rglob("*") if p.is_file()):
        h.update(f.relative_to(TEMPLATE).as_posix().encode())
        h.update(f.read_bytes())
    h.update(text.encode())
    h.update(f"{U}".encode())
    return h.hexdigest()[:16]


def run_case(H, e, N, U, run_dir):
    case = Path(run_dir).expanduser() / case_name(H, e, N, U)
    saved = case / "result.json"
    text, cells = block_mesh_dict(H, e, N)
    key = inputs_hash(text, U)
    if saved.exists():              # 입력이 같으면 다시 돌리지 않는다
        old = json.loads(saved.read_text())
        if old.get("inputs", key) == key:   # 지문이 없는 옛 결과는 그대로 쓴다
            return old
        stale = case.parent / "_stale" / f"{case.name}_{old['inputs']}"
        shutil.rmtree(stale, ignore_errors=True)
        stale.parent.mkdir(exist_ok=True)
        shutil.move(case, stale)    # 입력이 바뀐 옛 결과는 옮겨 두고 새로 해석
    if case.exists():               # 중간에 멈춘 케이스는 지우고 새로 시작
        shutil.rmtree(case)
    shutil.copytree(TEMPLATE, case)

    record = {"case": case.name, "H": H, "e": e, "N": N, "U": U,
              "status": None, "error": None, "cells": None, "inputs": key}
    start = time.time()
    try:
        # 1) 변수 치환: 형상은 blockMeshDict를 새로 쓰고, 유속은 0/U를 고친다
        (case / "system" / "blockMeshDict").write_text(text)
        run(["foamDictionary", "0/U", "-entry", "boundaryField/inlet/value",
             "-set", f"uniform ({U} 0 0)"], case, "foamDictionary")
        # 2) 격자 생성·검사와 해석
        run(["blockMesh"], case, "blockMesh")
        run(["checkMesh"], case, "checkMesh")
        if "Mesh OK" not in (case / "log.checkMesh").read_text():
            raise RuntimeError("checkMesh 격자 품질 검사 불합격")
        record["cells"] = cells
        run(["foamRun"], case, "foamRun")
        # 3) 결과 추출 (책 03-3)
        record.update(extract(case))
        record["status"] = "ok" if record["converged"] else "not_converged"
    except InterruptedError:        # 사용자가 멈춘 것은 실패가 아니므로
        raise                       # result.json을 남기지 않는다 (책 04-3)
    except Exception as err:        # 실패해도 기록만 하고 다음으로 넘어간다
        record["status"] = "failed"
        record["error"] = str(err)
    record["seconds"] = round(time.time() - start, 1)
    saved.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


if __name__ == "__main__":
    if shutil.which("foamRun") is None:
        sys.exit("OpenFOAM 환경이 없습니다: . /opt/openfoam14/etc/bashrc")
    H, e, N, U = map(float, sys.argv[1:5])
    run_dir = sys.argv[5] if len(sys.argv) > 5 else "~/cfd-runs/auto"
    r = run_case(H, e, int(N), U, run_dir)
    print(json.dumps(r, ensure_ascii=False, indent=2))
