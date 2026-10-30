# 11장. Active Learning으로 AI 모델 개선하기

앙상블 MLP로 예측 불확실성을 추정하고, 불확실한 설계를 골라 CFD를 더 돌려
대리모델을 다시 학습합니다. 같은 CFD 비용에서 무작위로 데이터를 더할 때와
정확도를 비교하고, 10장 최적점이 몰린 쪽으로 설계 범위를 넓혀 봅니다.
CPU만 씁니다. CFD를 돌리는 스크립트(new_test.py, al_loop.py, expand.py)는
OpenFOAM 14 환경이 필요합니다(책 02-1).

| 파일 | 책 | 내용 |
|---|---|---|
| common.py | 11장 공통 | 앙상블 MLP(한꺼번에 학습), 불확실성, 후보 풀, run_batch.py로 CFD 실행 |
| new_test.py | 11-1, 11-5 | 새 시험 데이터 200개(범위 안), --wide: 넓힌 범위 100개 |
| new_test.csv, wide_test.csv | 11-1, 11-5 | 위 두 시험 데이터의 CFD 결과 (다시 돌릴 필요 없음) |
| uncertainty.py | 11-1 | 앙상블·MC 드롭아웃·GP의 불확실성과 실제 오차 비교, 보정 배율 |
| select_samples.py | 11-2 | 무작위·불확실성 상위·불확실성+다양성 선택 비교 |
| al_loop.py | 11-3, 11-4 | 능동학습 반복 루프 (학습 -> 평가 -> 선택 -> CFD -> 추가) |
| compare.py | 11-4, 11-5 | 선택 방법별 기록 비교, 데이터 수-오차 곡선 |
| expand.py | 11-5 | 넓힌 범위의 수렴 지도, 넓힌 모델로 다시 최적화, CFD 재해석 |

```bash
. /opt/openfoam14/etc/bashrc
uv run python ch11/new_test.py              # CFD 200건 (동시 6개, 약 6분)
uv run python ch11/uncertainty.py           # ~/cfd-runs/ch11/uncertainty.png
uv run python ch11/select_samples.py        # ~/cfd-runs/ch11/select.png
for st in random top diverse; do            # 시드마다 세 방법을 동시에
    uv run python ch11/al_loop.py --strategy $st --seed 0 > al_$st.log &
done; wait
uv run python ch11/compare.py               # ~/cfd-runs/ch11/al_curve.png
uv run python ch11/expand.py                # 넓힌 범위에서 다시 최적화 (약 5분)
```

- al_loop.py는 CFD를 동시에 2개(--workers)씩 돌리고 PyTorch 스레드를 1개로
  씁니다. 세 방법을 함께 돌리면 물리 코어 6개를 거의 다 씁니다.
- 11-5의 실험은 --n0 418 --rounds 4로 train 전체에서 100개를 더하고,
  --space wide로 넓힌 범위의 새 영역에서 고릅니다. 명령은 책 11-5를 보세요.
- CFD 결과는 ~/cfd-runs/ch11 아래에 남고 git에는 넣지 않습니다.
  케이스 폴더에 result.json이 있으면 다시 해석하지 않으므로, 같은 명령을
  다시 실행하면 학습만 다시 합니다.
