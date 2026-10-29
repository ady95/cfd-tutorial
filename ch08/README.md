# 8장. NVIDIA PhysicsNeMo 활용하기

PhysicsNeMo의 공식 예제를 실행하고, 7장 물리장 데이터에 PhysicsNeMo FNO를 학습해
neuraloperator FNO, 5장의 숫자 모델과 비교합니다. OpenFOAM 격자를 그래프로 바꿔
MeshGraphNet의 비용도 잽니다. GPU가 필요합니다(이 책은 RTX 3090 24GB).

환경은 두 가지를 씁니다.

- uv 환경: pyproject.toml에 nvidia-physicsnemo가 들어 있습니다. uv sync만 하면 됩니다.
- PhysicsNeMo 컨테이너(nvcr.io/nvidia/physicsnemo/physicsnemo:26.08, 30.6 GB):
  공식 예제와 MeshGraphNet(torch_scatter 필요)에 씁니다.

| 파일 | 책 | 환경 | 내용 |
|---|---|---|---|
| check_darcy.py | 08-2 | 컨테이너 | 공식 darcy_fno 예제가 저장한 .mdlus를 불러와 새 표본으로 상대 오차 확인 |
| compare_arch.py | 08-3 | uv | neuraloperator FNO와 PhysicsNeMo FNO의 부분별 파라미터 수, 패딩 뒤 크기 |
| train_pnfno.py | 08-3 | uv | 7장과 같은 데이터·조건으로 PhysicsNeMo FNO 학습, pnfno.mdlus와 pnfno.json 저장 |
| pnfno_model.py | 08-3, 08-4 | uv | 저장한 PhysicsNeMo FNO 불러오기 |
| evaluate_pnfno.py | 08-3 | uv | 두 FNO의 split별 물리장 오차, 다시 계산한 값, 추론 시간, 오차 지도 |
| compare_all.py | 08-4 | uv | XGBoost·MLP·FNO 두 가지의 세 값 오차, 학습·추론 시간 |
| mesh_graph.py | 08-5 | 컨테이너 | OpenFOAM 격자 -> 그래프, MeshGraphNet 파라미터 수와 학습 스텝 비용 |

7장의 ~/cfd-runs/ch07/fields_1025x49.npz와 fno.pt가 있어야 합니다.

```bash
uv run python ch08/compare_arch.py
uv run python ch08/train_pnfno.py           # 약 11분 (RTX 3090)
uv run python ch08/evaluate_pnfno.py        # ~/cfd-runs/ch08/pnfno-error-map.png
uv run python ch08/compare_all.py           # 약 1분
```

컨테이너 실행 (08-2는 PhysicsNeMo 저장소 v2.2.0의 examples/cfd/darcy_fno 폴더에서):

```bash
img=nvcr.io/nvidia/physicsnemo/physicsnemo:26.08
opt="--rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
     --user $(id -u):$(id -g) -e HOME=/tmp"
docker run $opt -v $PWD:/work -w /work $img \
    python train_fno_darcy.py training.max_pseudo_epochs=16     # 약 33분
docker run $opt -v $PWD:/work -w /work -v ~/cfd-tutorial:/tutorial:ro $img \
    python /tutorial/ch08/check_darcy.py
docker run $opt -v ~/cfd-tutorial:/tutorial:ro \
    -v ~/cfd-runs/dataset:/dataset:ro $img python /tutorial/ch08/mesh_graph.py
```

학습한 모델(약 17 MB)은 저장소에 넣지 않았습니다. 위 명령으로 다시 만들 수 있습니다.
