# 4장. AI 학습용 CFD 데이터셋 생성하기

3장의 자동화 스크립트(ch03/run_case.py)로 설계 변수 조합 수백 개를 해석해
대리모델 학습용 데이터셋을 만듭니다. OpenFOAM 14와 bash 환경이 필요합니다.

| 파일 | 책 | 내용 |
|---|---|---|
| sampling_compare.py | 04-1 | 격자·무작위·LHS 샘플링 비교 (그림과 4차원 지표) |
| make_samples.py | 04-2 | 설계 변수 샘플 목록 생성 (범위 안 600 + 범위 밖 100, 시드 2026) |
| samples.csv | 04-2 | make_samples.py 결과. id, split, H, e, N, U, e_ratio |
| run_batch.py | 04-3 | samples.csv를 동시에 여러 개 해석 (기본 6개), 중단 후 이어 돌리기 |
| check_cases.py | 04-4 | 학습 사용 여부 자동 판정, 미수렴 흔들림, 반복 상한 비교, 실패 패턴 그림 |
| rerun_slow.py | 04-4 | 값이 안정된 미수렴 케이스만 반복 상한 20000으로 재해석 |
| build_dataset.py | 04-5 | 판정·재해석을 반영해 data/cooling_v1에 Parquet·CSV·meta.json 저장 |

```bash
uv run python ch04/sampling_compare.py      # ~/cfd-runs/sampling
uv run python ch04/make_samples.py          # ch04/samples.csv 다시 만들기
uv run python ch04/run_batch.py             # ~/cfd-runs/dataset (700개)
uv run python ch04/check_cases.py
uv run python ch04/rerun_slow.py            # ~/cfd-runs/dataset/rerun
uv run python ch04/build_dataset.py         # data/cooling_v1
```

해석을 다시 돌리지 않아도 5장부터 따라 할 수 있도록, 결과 데이터셋
data/cooling_v1(700행)을 저장소에 넣어 두었습니다. 만든 환경과 파일 지문은
data/cooling_v1/meta.json에 있습니다.
