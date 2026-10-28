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

```bash
uv run python ch05/explore.py               # ~/cfd-runs/ch05/explore.png
uv run python ch05/baseline.py              # ~/cfd-runs/ch05/baseline-residual.png
uv run python ch05/train_xgb.py             # ~/cfd-runs/ch05/learning-curve.png (약 1분)
```
