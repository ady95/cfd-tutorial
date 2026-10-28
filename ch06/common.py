"""책 6장 공통: 5장의 모델을 다시 만들고, 평가 지표를 계산한다.

- 다항 회귀(05-2), XGBoost(05-3), MLP(05-4)를 같은 설정으로 학습하는 함수
- models/surrogate_v1에 저장한 최종 MLP를 불러오는 함수
- MAE, RMSE, R², 상대 오차를 한 번에 계산하는 함수
모든 모델은 "입력 배열 -> 원래 단위의 출력 배열" 함수로 돌려준다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ch05"))
from data import (INPUTS, TARGETS, Scaler, from_model, load,  # noqa: E402,F401
                  to_model)

MODEL_DIR = ROOT / "models" / "surrogate_v1"
XGB_PARAMS = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.05,
              "subsample": 0.8, "seed": 0}
XGB_ROUNDS = [1819, 2393, 1718]     # 05-3에서 교차 검증으로 정한 트리 수


def fit_mean(x, y):
    """train 평균을 언제나 내놓는 기준선 (05-2)."""
    m = to_model(y).mean(axis=0)
    return lambda x_new: from_model(np.tile(m, (len(x_new), 1)))


def fit_poly(x, y, degree):
    """d차 다항 회귀 (05-2)."""
    p = make_pipeline(StandardScaler(), PolynomialFeatures(degree),
                      LinearRegression()).fit(x, to_model(y))
    return lambda x_new: from_model(p.predict(x_new))


def fit_xgb(x, y, rounds=XGB_ROUNDS):
    """출력마다 XGBoost 하나씩 (05-3 설정)."""
    z = to_model(y)
    boosters = [xgb.train(XGB_PARAMS, xgb.DMatrix(x, label=z[:, i]), n)
                for i, n in enumerate(rounds)]
    return lambda x_new: from_model(np.column_stack(
        [b.predict(xgb.DMatrix(x_new)) for b in boosters]))


def build_mlp(act="SiLU", width=64, depth=3):
    layers, n = [], len(INPUTS)
    for _ in range(depth):
        layers += [nn.Linear(n, width), getattr(nn, act)()]
        n = width
    return nn.Sequential(*layers, nn.Linear(n, len(TARGETS)))


def mlp_predictor(model, sx, sz):
    def predict(x_new):
        xs = torch.tensor(sx.transform(x_new), dtype=torch.float32)
        with torch.no_grad():
            z = model(xs).numpy().astype(float)
        return from_model(sz.inverse(z))
    return predict


def fit_mlp(x, y, seed=0, epochs=20000):
    """05-4와 같은 MLP를 CPU에서 full batch로 학습한다 (약 30초)."""
    sx, sz = Scaler().fit(x), Scaler().fit(to_model(y))
    xt = torch.tensor(sx.transform(x), dtype=torch.float32)
    zt = torch.tensor(sz.transform(to_model(y)), dtype=torch.float32)
    torch.manual_seed(seed)
    model = build_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad()
        ((model(xt) - zt) ** 2).mean().backward()
        opt.step()
        sched.step()
    return mlp_predictor(model, sx, sz)


def load_mlp_parts(model_dir=MODEL_DIR):
    """05-5에서 저장한 최종 MLP의 신경망과 입력·출력 표준화를 불러온다."""
    meta = json.loads((Path(model_dir) / "mlp.json").read_text())
    c = meta["config"]
    model = build_mlp(c["act"], c["width"], c["depth"])
    model.load_state_dict(torch.load(Path(model_dir) / "mlp.pt"))
    sx, sz = Scaler(), Scaler()
    sx.mean, sx.std = (np.array(meta["x_scaler"][k]) for k in ("mean", "std"))
    sz.mean, sz.std = (np.array(meta["z_scaler"][k]) for k in ("mean", "std"))
    return model, sx, sz


def load_mlp(model_dir=MODEL_DIR):
    """최종 MLP를 "입력 -> 원래 단위의 출력" 함수로 불러온다."""
    return mlp_predictor(*load_mlp_parts(model_dir))


def metrics(y_true, y_pred):
    """출력별 MAE, RMSE, R², 평균·최대 상대 오차 [%]를 dict로 돌려준다."""
    out = {}
    for i, t in enumerate(TARGETS):
        a, p = y_true[:, i], y_pred[:, i]
        err = p - a
        out[t] = {"MAE": np.abs(err).mean(),
                  "RMSE": np.sqrt((err ** 2).mean()),
                  "R2": 1 - (err ** 2).sum() / ((a - a.mean()) ** 2).sum(),
                  "rel_mean": (np.abs(err / a) * 100).mean(),
                  "rel_max": (np.abs(err / a) * 100).max()}
    return out
