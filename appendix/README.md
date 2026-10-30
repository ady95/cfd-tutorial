# 부록 실습

책의 부록에서 쓰는 코드입니다. 모든 명령은 저장소 맨 위 폴더에서 실행합니다. OpenFOAM을 쓰는 명령 앞에는 `. /opt/openfoam14/etc/bashrc`가 필요합니다.

## A-1 실행환경 고정·재현성 점검 (a1_env)

```bash
uv run python appendix/a1_env/check_env.py        # 책과 같은 환경인지 점검
bash appendix/a1_env/ref_check.sh                 # 기준 CFD·대리모델 값 재현 (약 30초)
docker build -f appendix/a1_env/Dockerfile -t cfd-tutorial:book .   # 약 13분, 9GB
docker run --rm --gpus all --user $(id -u):$(id -g) -v $PWD:/work \
    cfd-tutorial:book bash appendix/a1_env/ref_check.sh
```

## A-4 3D 해석과 복합 열전달 (a4_3d_cht)

```bash
uv run python appendix/a4_3d_cht/channel3d.py ~/cfd-runs/a4/3d 6 20        # 약 2분 (코어 6개)
uv run python appendix/a4_3d_cht/channel3d.py ~/cfd-runs/a4/3d_nz40 6 40   # 약 5분
uv run python appendix/a4_3d_cht/cht2d.py ~/cfd-runs/a4/cht al             # 판 없음 + 알루미늄 판, 약 4분
uv run python appendix/a4_3d_cht/cht2d.py ~/cfd-runs/a4/cht steel          # 판 없음 + 스테인리스 판
uv run python appendix/a4_3d_cht/plot_a4.py                                # ~/cfd-runs/a4/a4-3d-cht.png
```

## A-5 MeshGraphNet (a5_mgn)

PhysicsNeMo 컨테이너(8장과 같은 26.08)에서 실행합니다. 4장의 해석 결과 폴더(~/cfd-runs/dataset)가 필요합니다.

```bash
img=nvcr.io/nvidia/physicsnemo/physicsnemo:26.08
opt="--rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
     --user $(id -u):$(id -g) -e HOME=/tmp"
mkdir -p ~/cfd-runs/a5
run="docker run $opt -v $HOME/cfd-tutorial:/tutorial:ro
     -v $HOME/cfd-runs/dataset:/dataset:ro -v $HOME/cfd-runs/a5:/out
     -w /tutorial/appendix/a5_mgn $img"
$run python graphs.py /out/graphs.pt                  # 약 1분, 2.1 GB
$run python train_mgn.py /out/graphs.pt --check       # 셀 값으로 세 값 다시 계산
$run python train_mgn.py /out/graphs.pt /out 150      # 약 1.5시간 (RTX 3090)
$run python evaluate_mgn.py /out/graphs.pt /out       # ~/cfd-runs/a5/mgn-case.png
```

## A-6 구조 CAE 대리모델 (a6_structure)

```bash
python3 appendix/a6_structure/plate_case.py 0.25 1.0 0.3 ~/cfd-runs/a6/one   # 케이스 하나
python3 appendix/a6_structure/grid_check.py           # 격자 검사, 약 2분
uv run python appendix/a6_structure/run_plates.py     # 160개, 약 12분 -> plates.csv
uv run python appendix/a6_structure/train_plate.py    # ~/cfd-runs/a6/plate-kt.png
```

plates.csv(160행)는 저장소에 들어 있어 해석 없이 학습부터 할 수 있습니다.

## A-7 DrivAerML (a7_drivaerml)

DrivAerML(Ashton 외 2024, CC-BY-SA-4.0)의 요약 파일만 내려받습니다. 데이터를 쓸 때는 데이터셋 논문(arXiv 2408.11969)을 인용하세요.

```bash
uv run python appendix/a7_drivaerml/get_data.py      # 약 190 KB -> ~/cfd-runs/a7
uv run python appendix/a7_drivaerml/train_cd.py      # 약 1분, ~/cfd-runs/a7/drivaerml-curve.png
```
