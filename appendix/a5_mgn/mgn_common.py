"""책 부록 A-5 공통: 그래프 정규화와 셀 값에서 세 값(dp, dT_max, dT_avg) 계산."""
import numpy as np
import torch
from torch_geometric.data import Batch, Data

GRAD_T = 16666.67           # 가열면 온도 기울기 q/k [K/m] (03-2)
POWER = torch.tensor([1, 1, 2, 0])  # Ux, Uy, p, dT를 나눌 (U/0.1)의 지수


def flow_factor(U):
    """07-3과 같은 무차원화: Ux, Uy는 U로, p는 U²로 나눈다."""
    return (U / 0.1) ** POWER.float()


def fit_rms(graphs):
    """train 그래프의 무차원 정답을 채널마다 RMS로 나눌 값."""
    sq = torch.zeros(4)
    n = 0
    for g in graphs:
        y = g["y"] / flow_factor(g["U"])
        sq += (y ** 2).sum(0)
        n += len(y)
    return (sq / n).sqrt()


def to_data(g, rms, device):
    """그래프 dict -> PyG Data (정규화한 정답 포함)."""
    y = g["y"] / flow_factor(g["U"]) / rms
    return Data(x=g["x"], edge_attr=g["e"], edge_index=g["edge_index"],
                y=y, num_nodes=len(g["x"])).to(device)


def batch(datas):
    return Batch.from_data_list(datas)


def denorm(z, g, rms):
    """모델 출력 -> 물리 단위 셀 값 (Ux, Uy [m/s], p [m²/s²], dT [K])."""
    return z.cpu() * rms * flow_factor(g["U"])


def scalars(cells, g):
    """셀 값 (n, 4) -> (dp, dT_max, dT_avg).

    dp: x=5 mm 단면 셀의 평균 압력 (출구 압력 0)
    dT_max, dT_avg: 가열면에 붙은 셀 값에 기울기 x 거리를 더한 벽 온도의
                    최고와 면 길이 가중 평균
    """
    cells = cells.numpy() if torch.is_tensor(cells) else cells
    dp = cells[g["section"].numpy(), 2].mean()
    wall = cells[g["heated"].numpy(), 3] + GRAD_T * g["heated_dy"].numpy()
    w = g["heated_len"].numpy()
    return np.array([dp, wall.max(), (wall * w).sum() / w.sum()])


def rel_l2(pred, true):
    """채널마다 상대 L2 오차 [%] (셀 하나를 같은 무게로)."""
    return ((pred - true).norm(dim=0) / true.norm(dim=0) * 100).numpy()
