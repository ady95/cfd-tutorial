"""책 07-4·07-5 공통: 저장한 FNO를 불러와 물리장을 예측하고 오차를 잰다."""
import json

import numpy as np
import torch
from neuralop.models import FNO

from fno_data import FieldScaler, make_inputs


def load_fno(path):
    """train_fno.py가 저장한 fno.pt와 fno.json으로 (모델, 정규화, 설정)."""
    meta = json.loads(path.with_suffix(".json").read_text())
    model = FNO(n_modes=tuple(meta["modes"]), in_channels=2, out_channels=4,
                hidden_channels=meta["width"], n_layers=meta["layers"],
                domain_padding=meta["padding"]).cuda()
    model.load_state_dict(torch.load(path.with_suffix(".pt")))
    scaler = FieldScaler(meta["flow"])
    scaler.rms = np.array(meta["rms"], np.float32)
    return model.eval(), scaler, meta


def predict(model, scaler, mask, U, batch=16):
    """마스크 (n, ny, nx)와 유속 (n,) -> 원래 단위 물리장 (n, 4, ny, nx)."""
    x = torch.as_tensor(make_inputs(mask, U), device="cuda")
    with torch.no_grad():
        z = torch.cat([model(x[i:i + batch])
                       for i in range(0, len(x), batch)]).cpu().numpy()
    return scaler.inverse(z, U) * mask[:, None]


def rel_l2(pred, true, mask):
    """케이스·채널마다 유체 격자점의 상대 L2 오차 [%] -> (n, 4)."""
    m = mask[:, None].astype(bool)
    num = np.sqrt((((pred - true) * m) ** 2).sum(axis=(2, 3)))
    return num / np.sqrt(((true * m) ** 2).sum(axis=(2, 3))) * 100
