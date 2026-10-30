"""책 부록 A-5: cooling_v1의 사용 가능한 케이스 전체를 그래프 파일 하나로 저장한다.

08-5의 mesh_graph.py와 같은 방식으로 OpenFOAM 격자를 그래프로 바꾼다.
- 노드: 셀 하나. 특징 11개
    x/80, y/3 [mm 기준], U/0.1, H/2.5, e_ratio/0.4, N/15,
    붙어 있는 경계 표시 5개 (heatedWall, ribWall, 그 밖의 벽, inlet, outlet)
- 에지: 내부 면 하나가 잇는 두 셀 (양방향). 특징은 (dx, dy, 거리) / 0.1 mm
- 정답: 셀 값 Ux, Uy, p, dT (보간하지 않은 OpenFOAM 값)
- 지표용: x=5 mm 단면 셀, 가열면에 붙은 셀과 벽까지 거리·면 길이

설계 변수(H, e_ratio, N)를 모든 노드에 넣는 이유는 본문 참고.
메시지 전달 15번으로는 채널 길이(셀 600개 안팎)만큼 정보가 건너가지 못한다.

사용법 (PhysicsNeMo 컨테이너 안, 저장소 /tutorial, 해석 폴더 /dataset):
    python /tutorial/appendix/a5_mgn/graphs.py /out/graphs.pt
"""
import csv
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

DATASET = Path("/dataset")
CASES = Path(__file__).resolve().parents[2] / "data/cooling_v1/cases.csv"
T_IN = 293.15
GRAD_T = 16666.67           # 가열면 온도 기울기 q/k [K/m] (03-2)
FLAGS = [["heatedWall"], ["ribWall"], ["upperWall", "lowerWall"],
         ["inlet"], ["outlet"]]


def label_list(path):
    """OpenFOAM ASCII labelList (owner, neighbour) -> 정수 배열."""
    text = path.read_text()
    body = text.split("\n(\n", 1)[1].rsplit("\n)", 1)[0]
    return np.array(body.split(), dtype=np.int64)


def patches(poly):
    """boundary 파일 -> {이름: (startFace, nFaces)}."""
    text = (poly / "boundary").read_text()
    out = {}
    for name, body in re.findall(r"\n\s*(\w+)\s*\n\s*\{([^}]*)\}", text):
        n = re.search(r"nFaces\s+(\d+)", body)
        s = re.search(r"startFace\s+(\d+)", body)
        if n and s:
            out[name] = (int(s.group(1)), int(n.group(1)))
    return out


def to_graph(args):
    path, row = args
    poly = path / "constant/polyMesh"
    owner = label_list(poly / "owner")
    neigh = label_list(poly / "neighbour")
    bnd = patches(poly)

    foam = path / "case.foam"
    if not foam.exists():
        foam.touch()
    reader = pv.POpenFOAMReader(str(foam))
    reader.set_active_time_value(reader.time_values[-1])
    mesh = reader.read()["internalMesh"]
    xy = mesh.cell_centers().points[:, :2] * 1e3              # mm
    n = len(xy)

    flags = np.zeros((n, len(FLAGS)), np.float32)
    for k, names in enumerate(FLAGS):
        for name in names:
            s, m = bnd[name]
            flags[owner[s:s + m], k] = 1
    H, e_ratio, N, U = (float(row[k]) for k in ("H", "e_ratio", "N", "U"))
    design = np.array([U / 0.1, H / 2.5, e_ratio / 0.4, N / 15], np.float32)
    nodes = np.column_stack([xy[:, 0] / 80, xy[:, 1] / 3,
                             np.tile(design, (n, 1)), flags])

    inner = owner[:len(neigh)]
    edge = np.concatenate([np.stack([inner, neigh]),
                           np.stack([neigh, inner])], axis=1)
    d = xy[edge[1]] - xy[edge[0]]
    edges = np.column_stack([d, np.linalg.norm(d, axis=1)]) / 0.1

    c = mesh.cell_data
    target = np.column_stack([c["U"][:, 0], c["U"][:, 1], c["p"],
                              c["T"] - T_IN])

    # 지표용 셀: 가열면에 붙은 셀(벽까지 거리 = 셀 중심 y), x=5 mm 단면 셀
    s, m = bnd["heatedWall"]
    heated = owner[s:s + m]
    xs = np.array([np.ptp(mesh.get_cell(i).points[:, 0]) * 1e3
                   for i in heated])                        # 면 길이 [mm]
    # x=5 mm 단면: 입구 구간은 셀 폭이 일정하므로 셀 중심 x가 5 mm에
    # 가장 가까운 한 줄을 고른다
    col = np.abs(xy[:, 0] - 5.0)
    sec = np.where(col <= col.min() + 1e-6)[0]
    return {
        "case": row["case"], "split": row["split"],
        "x": torch.from_numpy(nodes.astype(np.float32)),
        "e": torch.from_numpy(edges.astype(np.float32)),
        "edge_index": torch.from_numpy(edge),
        "y": torch.from_numpy(target.astype(np.float32)),
        "U": U,
        "heated": torch.from_numpy(heated),
        "heated_dy": torch.from_numpy(xy[heated, 1] * 1e-3),   # m
        "heated_len": torch.from_numpy(xs),
        "section": torch.from_numpy(sec),
        "cfd": [float(row[k]) for k in ("dp", "dT_max", "dT_avg")],
    }


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/out/graphs.pt")
    rows = [r for r in csv.DictReader(CASES.open(encoding="utf-8"))
            if r["valid"] == "True"]
    jobs = [(DATASET / "rerun" / r["case"] if r["rerun"] == "True"
             else DATASET / r["case"], r) for r in rows]
    start = time.perf_counter()
    with ProcessPoolExecutor(6) as pool:
        graphs = list(pool.map(to_graph, jobs, chunksize=4))
    torch.save(graphs, out)
    nodes = np.array([len(g["x"]) for g in graphs])
    edges = np.array([g["edge_index"].shape[1] for g in graphs])
    print(f"{len(graphs)}개 그래프 {time.perf_counter() - start:.0f}초 -> {out}"
          f" ({out.stat().st_size / 1e9:.2f} GB)")
    print(f"노드 {nodes.min():,}~{nodes.max():,}개, 에지 {edges.min():,}~"
          f"{edges.max():,}개")
    split = {}
    for g in graphs:
        split[g["split"]] = split.get(g["split"], 0) + 1
    print("split별 개수:", split)
