"""책 9장 공통: 냉각 채널의 질량·에너지 보존을 격자 배열에서 계산한다.

2차원 채널의 단면(x가 같은 격자점 한 줄)마다
- 질량: 유량 Q(x) = ∫ Ux dy 는 입구 유량 U·H와 같아야 한다
- 에너지: 열 유량 E(x) = ∫ Ux·dT dy 는 그 단면보다 위쪽(상류)
  가열면에서 들어온 열 q/(ρ cp) x (가열 길이)와 같아야 한다
  (흐름 방향 열확산은 전체 가열량의 0.05% 이하라 뺀다, 09-2)

적분은 유체 격자점의 사다리꼴 공식에, 맨 위 격자점에서 벽(y = 위 벽 높이)까지
속도가 0으로 줄어드는 마지막 조각을 더한다. 위 벽 높이는 리브 아래에서
H - e, 나머지에서 H다. 단면별 가중치 W를 미리 만들어 두면 Q = Σ W·Ux로
계산되므로 PyTorch 손실에 그대로 쓸 수 있다.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ch03"))
sys.path.insert(0, str(ROOT / "ch07"))
from channel_mesh import L_HEAT, L_IN, rib_spans  # noqa: E402
from grid import axes  # noqa: E402

Q_RHO_CP = 1.0e4 / (998 * 4182)     # q/(ρ cp) = D x 온도 기울기 [K·m/s] (03-2)
E_TOTAL = Q_RHO_CP * L_HEAT         # 가열면 전체에서 들어온 열 [K·m/s·mm]


def top_height(H, e, N, nx):
    """단면마다 위 벽 높이 [mm] -> (nx,)."""
    x, _ = axes(nx, 2)
    top = np.full(nx, H)
    for a, b in rib_spans(N):
        top[(x > a + 1e-9) & (x < b - 1e-9)] = H - e
    return top


def weights(mask, H, e, N):
    """마스크 (ny, nx)와 형상 -> 단면 적분 가중치 W (ny, nx) [mm]."""
    ny, nx = mask.shape
    _, y = axes(nx, ny)
    dy = y[1] - y[0]
    top = top_height(H, e, N, nx)
    W = np.zeros((ny, nx))
    for j in range(nx):
        k = int(mask[:, j].sum()) - 1          # 맨 위 유체 격자점 번호
        W[:k + 1, j] = dy
        W[0, j] = W[k, j] = dy / 2
        W[k, j] += (top[j] - y[k]) / 2          # 벽까지 선형으로 0이 되는 조각
    return W.astype(np.float32)


def expected(H, U, nx):
    """단면마다 기대 유량 [mm·m/s]과 기대 열 유량 [K·m/s·mm] -> (nx,), (nx,)."""
    x, _ = axes(nx, 2)
    heated = np.clip(x - L_IN, 0, L_HEAT)
    return np.full(nx, U * H), Q_RHO_CP * heated


def all_weights(data, k=slice(None)):
    """npz 데이터의 케이스들(k)에 대한 W (n, ny, nx), 기대 유량·열 유량 (n, nx)."""
    H, er, N, U = data["inputs"][k].T
    masks = data["mask"][k]
    nx = masks.shape[2]
    W = np.stack([weights(m, h, h * r, int(n))
                  for m, h, r, n in zip(masks, H, er, N)])
    q, e = zip(*(expected(h, u, nx) for h, u in zip(H, U)))
    return W, np.stack(q).astype(np.float32), np.stack(e).astype(np.float32)


def residuals(Ux, dT, W, q_exp, e_exp):
    """단면별 질량 잔차 Q/Q_in - 1, 에너지 잔차 (E - E_기대)/E_TOTAL.

    numpy 배열과 torch 텐서 모두 받는다. 입력 (n, ny, nx) -> 출력 (n, nx).
    """
    Q = (W * Ux).sum(-2)
    E = (W * Ux * dT).sum(-2)
    return Q / q_exp - 1, (E - e_exp) / E_TOTAL


def to_torch(*arrays, device="cuda"):
    return [torch.as_tensor(a, device=device) for a in arrays]
