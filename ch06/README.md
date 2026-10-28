# 6장. AI 대리모델 성능 평가하기

5장의 모델을 지금까지 쓰지 않은 test와 범위 밖(OOD) 데이터로 평가하고,
데이터 나누기 방법, 계산 속도와 총비용을 비교합니다.
OpenFOAM 없이 uv 환경만 있으면 됩니다(grouped_samples.py로 만든 목록을
해석할 때만 OpenFOAM이 필요하며, 그 결과는 grouped_results.csv로 들어 있습니다).

| 파일 | 책 | 내용 |
|---|---|---|
| common.py | 6장 공통 | 5장 모델 다시 만들기, 최종 MLP 불러오기, MAE·RMSE·R²·상대 오차 |
| evaluate.py | 06-1 | 데이터셋 지문 확인, 최종 MLP의 val·test 지표, 지표 비교, 격자 오차와 비교 |
| grouped_samples.py | 06-2 | 형상 30개 x 유속 5개 샘플 목록 (grouped_samples.csv) |
| grouped_results.csv | 06-2 | 위 목록을 ch04/run_batch.py로 해석한 결과 |
| group_split.py | 06-2 | 무작위·형상 기준·운전조건 기준 교차 검증 비교 |
| ood.py | 06-3 | 범위 밖 오차, 경계에서 멀어질 때의 오차, OOD 감지 |
| speed.py | 06-4 | CFD와 대리모델 계산 시간, 총비용과 손익분기점 |
| compare.py | 06-5 | 모델 종합 비교표, out-of-fold 오차 분석과 그림 |

```bash
uv run python ch06/evaluate.py
uv run python ch06/group_split.py           # 약 3분 (CPU 6코어)
uv run python ch06/ood.py                   # ~/cfd-runs/ch06/ood.png
uv run python ch06/speed.py                 # 다른 계산이 없을 때 실행
uv run python ch06/compare.py               # ~/cfd-runs/ch06/error-by-input.png
```

06-2의 묶음 데이터를 직접 해석하려면 OpenFOAM 환경에서 다음을 실행합니다(약 4분).

```bash
uv run python ch06/grouped_samples.py
uv run python ch04/run_batch.py ~/cfd-runs/grouped \
    --samples ch06/grouped_samples.csv
```
