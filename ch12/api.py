"""책 12-2, 12-3: 냉각 채널 설계 예측·최적화·CFD 재검증 API (FastAPI).

GET  /info          모델 정보, 학습 범위, test 오차
POST /predict       설계 여러 개 -> 예측, 펌핑 동력, 추정 오차, 경고
POST /optimize      펌핑 동력 상한 -> 대리모델 최적 설계 (10-3)
POST /verify        설계 하나를 CFD로 재검증하는 작업을 시작 (10-5)
GET  /jobs/{job}    재검증 작업 상태와 결과, 보고서

모델은 환경 변수 CFD_MODEL의 폴더에서 읽는다. 없으면 pipeline.py가 만든
모델을, 그것도 없으면 5장 MLP(models/surrogate_v1)를 쓴다.

사용법: uv run uvicorn --app-dir ch12 api:app --port 8000
    /verify를 쓰려면 OpenFOAM 환경을 불러온 bash에서 실행한다.
"""
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import system

MODEL_DIR = Path(os.environ.get("CFD_MODEL") or system.default_model_dir())
sur = system.Surrogate(MODEL_DIR)
OUT = system.load_config(Path(__file__).parent / "config.toml")["out"]
JOBS = OUT / "jobs"                 # 작업 기록 (서버를 다시 켜도 남는다)
JOBS.mkdir(parents=True, exist_ok=True)
cfd_pool = ThreadPoolExecutor(1)    # CFD 작업은 한 번에 하나씩
app = FastAPI(title="냉각 채널 설계 API")


class Design(BaseModel):
    H: float = Field(gt=0, description="채널 높이 [mm]")
    e_ratio: float = Field(gt=0, lt=1, description="리브 높이 비율 e/H")
    N: int = Field(ge=1, le=59, description="리브 개수 (60mm에 폭 0.5mm)")
    U: float = Field(gt=0, description="입구 유속 [m/s]")

    def row(self):
        return [self.H, self.e_ratio, self.N, self.U]


class PredictRequest(BaseModel):
    designs: list[Design] = Field(min_length=1, max_length=100_000)


class OptimizeRequest(BaseModel):
    p_max_ratio: float = Field(1.0, gt=0, description="동력 상한 / 기준 설계")
    gap_min: float = Field(1.0, gt=0, description="최소 틈 [mm]")


class VerifyRequest(BaseModel):
    design: Design
    cells: list[int] = [30, 60]
    feedback: bool = True           # 결과를 다음 학습 데이터로 쌓을지


@app.get("/info")
def info():
    m = sur.meta
    return {"model": m["model"], "dir": str(MODEL_DIR),
            "members": sur.members, "input_range": m["input_range"],
            "dataset_sha256": m.get("dataset_sha256"),
            "evaluation": m.get("evaluation"), "k": sur.k}


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.perf_counter()
    rows = sur.describe([d.row() for d in req.designs])
    return {"results": rows,
            "warnings": sum(r["status"] == "warning" for r in rows),
            "ms": round((time.perf_counter() - start) * 1000, 2)}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    r = system.optimize(sur, req.p_max_ratio, req.gap_min)
    d = r["design"]
    r["result"] = sur.describe([[d[n] for n in system.NAMES]])[0]
    return r


def run_job(job, req):
    """CFD 재검증 작업 (스레드에서 실행). 결과와 보고서를 JOBS에 남긴다."""
    path = JOBS / f"{job}.json"
    rec = json.loads(path.read_text())
    rec["status"] = "running"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    try:
        res = system.verify(sur, req.design.model_dump(), OUT / "verify-api",
                            cells=tuple(req.cells), p_max_ratio=1.0)
        if req.feedback:
            res["feedback_added"] = system.add_feedback(
                OUT / "feedback.csv", res, OUT / "verify-api", req.cells[0])
        text = system.verify_report(res, sur.meta["model"])
        (JOBS / f"{job}.md").write_text(text, encoding="utf-8")
        rec.update(status="done", result=res, report=text)
    except Exception as err:        # 작업이 죽어도 서버는 계속 돈다
        rec.update(status="error", error=str(err))
    rec["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2))


@app.post("/verify", status_code=202)
def verify(req: VerifyRequest):
    job = uuid.uuid4().hex[:8]
    rec = {"job": job, "status": "queued", "request": req.model_dump(),
           "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
           "prediction": sur.describe([req.design.row()])[0]}
    (JOBS / f"{job}.json").write_text(json.dumps(rec, ensure_ascii=False,
                                                 indent=2))
    cfd_pool.submit(run_job, job, req)
    return {"job": job, "status": "queued"}


@app.get("/jobs/{job}")
def job_status(job: str):
    path = JOBS / f"{job}.json"
    if not job.isalnum() or not path.exists():
        raise HTTPException(404, f"작업 {job} 없음")
    return json.loads(path.read_text())
