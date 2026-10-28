# 2장. OpenFOAM으로 CFD 실습 시작하기

OpenFOAM 14와 bash 환경이 필요합니다(책 02-1). GPU는 쓰지 않습니다.

| 파일 | 책 | 내용 |
|---|---|---|
| inlet_turbulence.py | 02-2 | 입구 난류 값(k, epsilon)과 레이놀즈 수 계산 |
| convergence_check.sh | 02-2 | 수렴 기준을 끄고 2000회까지 돌려 입구 평균 압력 변화 확인 |
| channel2d/ | 02-3 | 2D 평행 평판 층류 채널 케이스 (H 2 mm, L 100 mm, 물, 0.1 m/s) |
| check_channel.py | 02-3 | channel2d 결과를 층류 이론해와 비교 |

```bash
uv run python ch02/inlet_turbulence.py
bash ch02/convergence_check.sh

cp -r ch02/channel2d ~/cfd-runs/channel2d
~/cfd-runs/channel2d/Allrun
uv run python ch02/check_channel.py ~/cfd-runs/channel2d
```
