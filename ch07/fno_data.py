"""책 07-3: 격자 배열 파일을 FNO 학습용 입력·출력 텐서로 바꾼다.

입력 채널 (케이스마다 49 x 1025 그림 2장)
- mask: 유체 1, 리브 속·채널 밖 0. H, e, N이 모두 이 그림에 담긴다
- U: 입구 유속을 0.1 m/s로 나눈 값을 유체 영역 전체에 채운 그림
x, y 좌표는 FNO가 스스로 2채널을 덧붙인다(positional_embedding="grid").

출력 채널: Ux, Uy, p, dT. 유속으로 무차원화한 뒤 채널마다 RMS로 나눈다.
"""
from pathlib import Path

import numpy as np
import torch

FIELDS_FILE = Path("~/cfd-runs/ch07/fields_1025x49.npz").expanduser()
U_REF = 0.1                 # 입구 유속 기준값 [m/s]


def load(path=FIELDS_FILE):
    """npz를 dict로 읽는다. 0.6 GB를 한 번에 메모리에 올린다."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def make_inputs(mask, U):
    """(n, 49, 1025) 마스크와 (n,) 유속 -> (n, 2, 49, 1025) 입력."""
    mask = mask.astype(np.float32)
    u = (np.asarray(U, np.float32) / U_REF)[:, None, None] * mask
    return np.stack([mask, u], axis=1)


class FieldScaler:
    """출력 채널을 학습하기 좋은 크기로 바꾼다.

    flow=True면 먼저 입구 유속으로 무차원화한다: Ux, Uy는 U로, p는 U²로
    나눈다(U는 U_REF로 나눈 값). 그다음 채널마다 train 유체 격자점의 RMS로
    나눈다. 평균은 빼지 않는다. 유체 밖의 0이 그대로 0으로 남아야 하기 때문이다.
    """
    POWER = np.array([1, 1, 2, 0])      # Ux, Uy, p, dT를 나눌 (U/U_REF)의 지수

    def __init__(self, flow=True):
        self.flow = flow

    def _factor(self, U):
        u = np.asarray(U, np.float32) / U_REF
        if not self.flow:
            u = np.ones_like(u)
        return u[:, None, None, None] ** self.POWER[None, :, None, None]

    def fit(self, fields, mask, U):
        f = fields / self._factor(U)
        m = mask.astype(bool)
        self.rms = np.array([np.sqrt((f[:, c][m] ** 2).mean())
                             for c in range(f.shape[1])], np.float32)
        return self

    def transform(self, fields, U):
        return fields / self._factor(U) / self.rms[None, :, None, None]

    def inverse(self, z, U):
        return z * self.rms[None, :, None, None] * self._factor(U)


def split_tensors(data, split, scaler, device="cuda", step=1):
    """split 하나의 (입력, 정규화 출력, 마스크) 텐서. step=2면 한 칸씩 건너뛴다."""
    k = data["split"] == split
    s = (slice(None), slice(None, None, step), slice(None, None, step))
    mask = data["mask"][k][s]
    U = data["inputs"][k, 3]
    x = make_inputs(mask, U)
    y = scaler.transform(data["fields"][k][:, :, ::step, ::step], U)
    as_t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
    return as_t(x), as_t(y), as_t(mask.astype(np.float32))
