"""책 12-4: 파이프라인 기록과 API 측정으로 시스템 전체의 비용을 정리한다.

- 단계별 실제 시간: pipeline.py가 남긴 state.json (단계마다 마지막 실행)
- 준비 비용: CFD·데이터셋·학습·평가 단계. CFD는 코어 시간(케이스 시간 합)도
- 설계 한 건 평가: CFD(동시 6개일 때 한 건당) 대 API 예측 -> 손익분기 건수 (06-4)
- 설계 질문 하나(제약 아래 최적 설계 찾기): 대리모델 최적화 대 CFD 직접
  BoTorch (10-5) -> 손익분기 질문 수. 두 방법 모두 CFD 재검증이 따로 든다.
  비용 기준을 섞지 않고 둘로 나눠 센다.
  CFD 코어 시간: 준비 CFD 코어 시간 / 질문 하나의 BoTorch CFD 시간.
    대리모델 최적화는 CFD를 쓰지 않는다. GP 계산·학습 같은 CFD 밖 계산은 넣지 않았다.
  기다리는 시간: 준비 실제 시간 / (BoTorch 실제 시간 - 대리모델 최적화 시간).
    준비는 6코어 동시, BoTorch는 코어 하나로 잰 경과 시간이라 같은 자원의 비교가 아니다.
  둘 다 질문마다 BoTorch를 처음부터 돌리고 앞 질문의 CFD를 다시 쓰지 않는다고 본다.

사용법: uv run python ch12/cost.py [설정 파일]   (pipeline.py, bench_api.py 뒤)
결과:   표 출력과 작업 폴더의 cost.png
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

import system

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

BO_CFD_MIN = [9.8, 9.5, 7.4]    # 10-5: CFD 직접 BoTorch 60회의 CFD 시간 [분]
BO_WALL_MIN = [10.8, 10.5, 8.2]  # 10-5: 같은 실행의 실제 걸린 시간 [분]

cfg = system.load_config(sys.argv[1] if len(sys.argv) > 1 else
                         Path(__file__).parent / "config.toml")
out = cfg["out"]
state = json.loads((out / "state.json").read_text())
bench = json.loads((out / "bench.json").read_text())

# 단계마다 마지막으로 실제 실행한 시간이다. 건너뛴 단계는 앞 실행의 값이 남는다
print(f"단계별 실제 시간 ({cfg['name']}, 마지막 실행)")
for s in ["samples", "cfd", "dataset", "train", "evaluate", "optimize",
          "verify", "report"]:
    print(f"  {s:<9} {state[s]['seconds']:7.1f}초  {state[s]['finished']}")

# 준비 비용: 모델이 쓸 수 있게 될 때까지
prep = sum(state[s]["seconds"] for s in ("cfd", "dataset", "train",
                                         "evaluate"))
cases = out / "cases"
batch = pd.read_csv(cases / "results.csv")
rerun = [json.loads(f.read_text())["seconds"]
         for f in (cases / "rerun").glob("*/result.json")]
core_h = (batch["seconds"].sum() + sum(rerun)) / 3600
log = (out / "logs" / "cfd.log").read_text()
batch_wall = float(re.search(r"걸린 시간 [\d.]+분 \((\d+)초\)", log)[1])
print(f"\n준비 비용: 실제 시간 {prep / 60:.1f}분"
      f" (CFD 배치 {batch_wall / 60:.1f}분, 재해석"
      f" {(state['cfd']['seconds'] - batch_wall) / 60:.1f}분,"
      f" 학습 {state['train']['seconds']:.0f}초)")
print(f"           CFD 코어 시간 {core_h:.2f}시간"
      f" (배치 {len(batch)}건 + 재해석 {len(rerun)}건)")

# 설계 한 건
cfd_one = batch_wall / len(batch)
api_one = bench["predict_ms"] / 1000
print(f"\n설계 한 건: CFD {cfd_one:.2f}초 (동시 6개일 때 한 건당),"
      f" API {bench['predict_ms']:.2f} ms")
print(f"  손익분기 {prep / (cfd_one - api_one):.0f}건"
      f" (준비 실제 시간 / 한 건 차이)")

# 설계 질문 하나
bo = sum(BO_CFD_MIN) / len(BO_CFD_MIN)
bo_wall = sum(BO_WALL_MIN) / len(BO_WALL_MIN)
opt = bench["optimize_s"] / 60
# 재검증: 격자마다 동시에 돌린 케이스 중 가장 긴 것의 합 (다시 실행하면 캐시라 0초)
ver = sum(max(json.loads(f.read_text())["seconds"]
              for f in (out / "verify" / f"nh{c}").glob("*/result.json"))
          for c in cfg["verify"]["cells"]) / 60
print(f"\n설계 질문 하나: 대리모델 최적화 {opt * 60:.1f}초 (CFD 없음)")
print(f"  CFD 직접 BoTorch: CFD {bo:.1f}분, 실제 {bo_wall:.1f}분"
      " (코어 하나, 10-5)")
print(f"  손익분기 {core_h * 60 / bo:.0f}개"
      f" (CFD 코어 시간: 준비 {core_h * 60:.0f}분 / 질문 하나 {bo:.1f}분)")
print(f"  손익분기 {prep / 60 / (bo_wall - opt):.0f}개"
      f" (기다리는 시간: 준비 {prep / 60:.1f}분 / 질문 하나 차이"
      f" {bo_wall - opt:.1f}분)")
print(f"  CFD 재검증 {ver:.1f}분은 두 방법 모두 질문마다 든다")

# 그림: 평가 건수와 질문 수에 따른 누적 비용 (재검증 비용은 양쪽 같아 뺐다)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
n = np.linspace(0, 2000, 201)
a.plot(n, n * cfd_one / 60, color="#7f7f7f", lw=2, label="CFD (6 parallel)")
a.plot(n, prep / 60 + n * api_one / 60, color="#2a6fdb", lw=2,
       label="surrogate (build + API)")
a.axvline(prep / (cfd_one - api_one), color="k", ls="--", lw=1)
a.set(xlabel="designs evaluated", ylabel="wall time [min]",
      title="One design at a time")
q = np.arange(0, 41)
b.plot(q, q * bo / 60, color="#7f7f7f", lw=2,
       label="CFD + BoTorch per question")
b.plot(q, np.full(len(q), core_h), color="#2a6fdb", lw=2,
       label="surrogate (dataset CFD only)")    # 최적화는 CFD를 쓰지 않는다
b.axvline(core_h * 60 / bo, color="k", ls="--", lw=1)
b.set(xlabel="design questions", ylabel="CFD core time [h]",
      title="Constrained optimum per question")
for ax in (a, b):
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(out / "cost.png", dpi=150)
print(f"\n그림: {out / 'cost.png'}")
