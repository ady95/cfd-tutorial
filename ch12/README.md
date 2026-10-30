# 12장. CFD–AI 통합 설계 자동화 시스템 완성하기

4~11장에서 만든 조각을 하나의 시스템으로 묶습니다. 설정 파일 하나로
샘플링 -> CFD -> 데이터셋 -> 학습 -> 평가 -> 최적화 -> CFD 재검증을 실행하고,
결과 모델을 FastAPI 예측 API로 띄웁니다. CPU만 씁니다. CFD를 돌리는 단계
(pipeline.py, API의 /verify)는 OpenFOAM 14 환경이 필요합니다(책 02-1).

| 파일 | 책 | 내용 |
|---|---|---|
| system.py | 12장 공통 | 설정 읽기, 대리모델 불러오기, 예측·경고·추정 오차, 최적화, CFD 재검증·판정, feedback.csv, 보고서 |
| config.toml | 12-1 | 파이프라인 기본 설정 (4~10장 설정 그대로, CFD 700건) |
| quick.toml | 12-1 | 흐름만 확인하는 작은 설정 (CFD 60건) |
| pipeline.py | 12-1, 12-3 | 단계별 실행, 건너뛰기, 품질 기준, 로그, 보고서 |
| api.py | 12-2, 12-3 | /info, /predict, /optimize, /verify, /jobs |
| bench_api.py | 12-2 | API 응답 시간 측정 |
| client.py | 12-3 | 최적화 요청부터 CFD 재검증 보고서까지 |
| cost.py | 12-4 | 파이프라인·API 기록으로 총비용과 손익분기점 정리 |

```bash
. /opt/openfoam14/etc/bashrc
uv run python ch12/pipeline.py ch12/quick.toml   # 약 2분, 품질 기준에서 멈춤
uv run python ch12/pipeline.py                   # 전체 (저자 환경 약 40분)
uv run uvicorn --app-dir ch12 api:app --port 8000 &
uv run python ch12/bench_api.py
uv run python ch12/client.py                     # 최적화 -> CFD 재검증
uv run python ch12/pipeline.py                   # 재검증 결과로 다시 학습
uv run python ch12/cost.py
```

- 결과는 ~/cfd-runs/ch12/<설정의 name> 아래에 쌓이고 git에는 넣지 않습니다.
  단계별 실행 기록은 state.json, 로그는 pipeline.log와 logs 폴더에 있습니다.
- 파이프라인을 돌리지 않았다면 API는 5장 MLP(models/surrogate_v1)로 뜹니다.
  이때는 추정 오차 없이 예측과 범위 경고만 돌려줍니다.
- train.members를 1로 바꾸면 11장의 앙상블 대신 MLP 하나를 학습합니다.
