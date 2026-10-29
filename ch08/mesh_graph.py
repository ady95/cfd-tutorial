"""책 08-5: OpenFOAM 격자를 그래프로 바꾸고 MeshGraphNet의 크기와 비용을 잰다.

격자를 균일 격자로 옮기지 않고 그대로 쓴다.
- 노드: 셀 하나. 특징은 셀 중심 (x, y)와 입구 유속 U
- 에지: 내부 면 하나가 이웃한 두 셀을 잇는다 (양방향). 특징은 (dx, dy, 거리)
- 정답: 셀 값 Ux, Uy, p, dT (보간하지 않은 OpenFOAM 값)
PhysicsNeMo의 MeshGraphNet은 PyTorch Geometric과 torch_scatter가 필요하다.
torch 2.14용 torch_scatter 휠이 없으므로 PhysicsNeMo 컨테이너에서 실행한다.

사용법 (컨테이너 안, 저장소를 /tutorial, 데이터셋을 /dataset으로 연결):
    python /tutorial/ch08/mesh_graph.py
결과:   케이스별 셀 수 분포, 그래프 하나의 크기,
        MeshGraphNet 파라미터 수, 한 스텝 학습 시간과 GPU 메모리
"""
import re
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from torch_geometric.data import Data

from physicsnemo.models.meshgraphnet import MeshGraphNet

DATASET = Path("/dataset")
T_IN = 293.15


def label_list(path):
    """OpenFOAM ASCII labelList (owner, neighbour) -> 정수 배열."""
    text = path.read_text()
    body = text.split("\n(\n", 1)[1].rsplit("\n)", 1)[0]
    return np.array(body.split(), dtype=np.int64)


def n_cells(case):
    head = (case / "constant/polyMesh/owner").read_text()[:1000]
    return int(re.search(r"nCells:\s*(\d+)", head).group(1))


def to_graph(case):
    """케이스 폴더 -> (노드 특징, 에지 특징, 에지 목록, 정답)."""
    poly = case / "constant/polyMesh"
    owner = label_list(poly / "owner")
    neigh = label_list(poly / "neighbour")
    owner = owner[:len(neigh)]                  # 앞쪽이 내부 면이다
    reader = pv.OpenFOAMReader(str(case / "case.foam"))
    reader.set_active_time_value(reader.time_values[-1])
    mesh = reader.read()["internalMesh"]
    xy = mesh.cell_centers().points[:, :2] * 1e3          # mm
    U_in = float(case.name.split("_U")[1])
    nodes = np.column_stack([xy, np.full(len(xy), U_in)])
    edge = np.concatenate([np.stack([owner, neigh]),
                           np.stack([neigh, owner])], axis=1)
    d = xy[edge[1]] - xy[edge[0]]
    edges = np.column_stack([d, np.linalg.norm(d, axis=1)])
    c = mesh.cell_data
    target = np.column_stack([c["U"][:, 0], c["U"][:, 1], c["p"],
                              c["T"] - T_IN])
    return nodes, edges, edge, target


cases = sorted(p for p in DATASET.iterdir() if (p / "result.json").exists())
cells = np.array([n_cells(p) for p in cases])
print(f"케이스 {len(cases)}개의 셀 수: 최소 {cells.min():,}, 중앙값 "
      f"{int(np.median(cells)):,}, 최대 {cells.max():,}")

# 설계 범위 안(H <= 2.5, U <= 0.15)에서 셀 수가 중앙값인 케이스
inside = [i for i, p in enumerate(cases)
          if float(p.name[1:].split("_")[0]) <= 2.5
          and float(p.name.split("_U")[1]) <= 0.15]
case = cases[inside[int(np.argsort(cells[inside])[len(inside) // 2])]]
start = time.perf_counter()
nodes, edges, edge, target = to_graph(case)
print(f"\n{case.name}: 노드 {len(nodes):,}개, 에지 {edge.shape[1]:,}개 "
      f"(그래프 만들기 {time.perf_counter() - start:.2f}초)")
print(f"노드 특징 {nodes.shape[1]}개 (x, y, U), 에지 특징 {edges.shape[1]}개 "
      f"(dx, dy, 거리), 정답 {target.shape[1]}개 (Ux, Uy, p, dT)")

dev = "cuda"
as_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
x, e, y = as_t(nodes), as_t(edges), as_t(target)
graph = Data(edge_index=torch.as_tensor(edge, device=dev), num_nodes=len(x))
model = MeshGraphNet(input_dim_nodes=3, input_dim_edges=3, output_dim=4).to(dev)
print(f"\nMeshGraphNet 기본 설정 (처리 층 15개, 폭 128): 파라미터 "
      f"{sum(p.numel() for p in model.parameters()):,}개")

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
torch.cuda.reset_peak_memory_stats()
for i in range(25):
    if i == 5:
        torch.cuda.synchronize()
        start = time.perf_counter()
    loss = torch.nn.functional.mse_loss(model(x, e, graph), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
torch.cuda.synchronize()
step = (time.perf_counter() - start) / 20
print(f"그래프 하나 학습 스텝 {step * 1e3:.1f} ms, GPU 최대 메모리 "
      f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
with torch.no_grad():
    for _ in range(5):
        model(x, e, graph)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        model(x, e, graph)
    torch.cuda.synchronize()
print(f"그래프 하나 추론 {(time.perf_counter() - start) / 20 * 1e3:.1f} ms")
