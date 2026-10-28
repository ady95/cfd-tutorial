# 5장. XGBoost·MLP 기반 대리모델 개발하기

4장에서 만든 데이터셋(data/cooling_v1)으로 설계 변수 네 개(H, e/H, N, U)에서
압력손실과 가열면 온도 상승을 예측하는 대리모델을 만듭니다.
OpenFOAM 없이 uv 환경만 있으면 됩니다.

| 파일 | 책 | 내용 |
|---|---|---|
| data.py | 05-1 | 데이터 읽기, 로그 변환, 표준화, 상대 오차 (5장 공통) |
| explore.py | 05-1 | 출력 분포·순위 상관계수와 그림 |
| baseline.py | 05-2 | 평균값·선형·거듭제곱·다항(2~6차) 회귀 기준선과 잔차 그림 |
| train_xgb.py | 05-3 | XGBoost 출력별 학습(교차 검증·조기 종료), CPU·GPU 시간, 특징 중요도, 학습 곡선 |
| train_mlp.py | 05-4 | PyTorch MLP(세 출력 함께), 활성화 함수·시드 비교, CPU·GPU 속도, 모델별 오차 분포 그림 |
| tune.py | 05-5 | Optuna로 XGBoost·MLP 하이퍼파라미터 탐색, 최종 MLP를 models/surrogate_v1에 저장 |

```bash
uv run python ch05/explore.py               # ~/cfd-runs/ch05/explore.png
uv run python ch05/baseline.py              # ~/cfd-runs/ch05/baseline-residual.png
uv run python ch05/train_xgb.py             # ~/cfd-runs/ch05/learning-curve.png (약 1분)
uv run python ch05/train_mlp.py             # ~/cfd-runs/ch05/mlp-training.png (약 3분)
uv run python ch05/tune.py                  # models/surrogate_v1, ~/cfd-runs/ch05/tuning.png
```

최종 대리모델(models/surrogate_v1)은 저장소에 들어 있습니다. mlp.pt는 가중치,
mlp.json은 구조·표준화 값·val 오차·학습에 쓴 데이터셋 지문입니다.
