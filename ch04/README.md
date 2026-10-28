# 4장. AI 학습용 CFD 데이터셋 생성하기

3장의 자동화 스크립트(ch03/run_case.py)로 설계 변수 조합 수백 개를 해석해
대리모델 학습용 데이터셋을 만듭니다. OpenFOAM 14와 bash 환경이 필요합니다.

| 파일 | 책 | 내용 |
|---|---|---|
| sampling_compare.py | 04-1 | 격자·무작위·LHS 샘플링 비교 (그림과 4차원 지표) |

```bash
uv run python ch04/sampling_compare.py      # ~/cfd-runs/sampling
```
