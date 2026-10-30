# 10장. AI 대리모델을 이용한 설계 최적화

5장의 최종 MLP(models/surrogate_v1)로 냉각 채널 설계를 최적화하고, 찾은 후보를
CFD로 다시 해석해 확인합니다. CPU만 씁니다. CFD를 돌리는 스크립트(define.py
--cfd, verify.py, bo_cfd.py)는 OpenFOAM 14 환경이 필요합니다(책 02-1).

| 파일 | 책 | 내용 |
|---|---|---|
| problem.py | 10장 공통 | 설계 변수 범위, 기준 설계, 틈 제약, 펌핑 동력, 대리모델·CFD 평가, 0~1 변수 변환, 제약 최적점 |
| define.py | 10-1 | 제약별로 남는 설계 공간, 학습 범위 밖 예측과 CFD 비교(--cfd) |
| objectives.py | 10-2 | 압력손실과 펌핑 동력 비교, 가중합 목적함수의 가중치별 최적 설계 |
| optimize.py | 10-3 | 무작위, Optuna TPE, Optuna GPSampler, BoTorch 단일 목적 비교 |
| bo.py | 10-3, 10-5 | BoTorch 제약 베이지안 최적화 (evaluate 함수만 바꿔 대리모델·CFD 공용) |
| pareto.py | 10-4 | 파레토 프런트, NSGA-II·TPE 다목적 비교, 후보 A~D 저장 |
| verify.py | 10-5 | 후보를 셀 30·60 격자 CFD로 재해석, 예측·격자 오차, 실제 개선량 |
| bo_cfd.py | 10-5 | 대리모델 없이 CFD를 직접 BoTorch로 최적화 |

```bash
uv run python ch10/define.py                # --cfd: CFD 8건 (약 4분)
uv run python ch10/objectives.py
uv run python ch10/optimize.py              # ~/cfd-runs/ch10/optimize.png
uv run python ch10/pareto.py                # pareto.png, candidates.json
. /opt/openfoam14/etc/bashrc
uv run python ch10/verify.py                # ~/cfd-runs/ch10/verify
for s in 0 1 2; do
    uv run python ch10/bo_cfd.py --seed $s > bo_cfd_s$s.log &
done; wait
```

- optimize.py는 GP 계산 때 PyTorch 스레드를 1개로 줄입니다. 점 100개 남짓의
  작은 행렬에서는 스레드가 많으면 오히려 4배 느렸습니다(10-3).
- Optuna GPSampler는 greenlet 패키지가 있으면 획득 함수 최적화를 묶어서
  빠르게 합니다. pyproject.toml에 넣어 두었습니다.
- bo_cfd.py는 시드마다 코어 하나로 CFD를 차례로 돌립니다. 시드 여러 개를
  동시에 돌려도 되지만, 서버의 다른 계산과 겹치면 시간이 달라집니다.
