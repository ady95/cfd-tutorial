# 2장. OpenFOAM으로 CFD 실습 시작하기

OpenFOAM 14와 bash 환경이 필요합니다(책 02-1). GPU는 쓰지 않습니다.

| 파일 | 책 | 내용 |
|---|---|---|
| inlet_turbulence.py | 02-2 | 입구 난류 값(k, epsilon)과 레이놀즈 수 계산 |
| convergence_check.sh | 02-2 | 수렴 기준을 끄고 2000회까지 돌려 입구 평균 압력 변화 확인 |

```bash
uv run python ch02/inlet_turbulence.py
bash ch02/convergence_check.sh
```
