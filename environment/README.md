# 실행 환경

책의 실습을 실행한 환경과 패키지 버전을 기록합니다.

## Python 패키지

저장소 루트의 `pyproject.toml`·`uv.lock`·`.python-version`이 기준입니다. `uv sync` 한 번으로 같은 버전이 설치됩니다. 장이 늘어나면 그 장에 필요한 패키지를 추가하고 잠금 파일을 갱신합니다.

| 패키지 | 버전 | 추가한 장 |
|---|---|---|
| Python | 3.12 | 02 |
| torch | 2.14.1 (cu130) | 02 |
| numpy | 2.5.4 | 02 |
| pyvista | 0.49.1 (vtk 9.7.1) | 02 |
| matplotlib | 3.11.2 | 02 |

### 화면 없는 서버에서 그림 만들기

PyPI의 vtk 9.7.1 휠은 화면(X 디스플레이)이 없으면 EGL로 렌더링합니다. NVIDIA 드라이버가 있는 서버라면 추가 설치 없이 PyVista로 PNG를 만들 수 있습니다. 이때 출력되는 `bad X server connection` 경고는 무시해도 됩니다.

### GPU 드라이버와 PyTorch CUDA 빌드

PyPI의 Linux용 torch 2.14.1 기본 빌드는 CUDA 13.0용(`+cu130`)이며 NVIDIA 드라이버 580 이상이 필요합니다. 드라이버가 더 오래되었다면 드라이버를 올리거나, PyTorch 공식 설치 안내에서 낮은 CUDA 빌드를 고르세요.

## OpenFOAM

Foundation 판 apt 패키지 `openfoam14`을 사용합니다. 설치 방법은 책 02-1을 참고하세요.

## 실측 환경

| 항목 | 내용 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| CPU / RAM | 12코어 / 62GB |
| GPU | RTX 3090 24GB, 드라이버 580.178.04 |
| OpenFOAM | openfoam14 20261008, Open MPI 4.1.2 |
| uv / Python | 0.13.1 / 3.12.15 |
