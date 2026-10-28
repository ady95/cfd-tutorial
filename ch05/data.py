"""책 5장 공통: cooling_v1 데이터셋을 입력·출력 배열로 읽고 변환한다.

05-1에서 정한 규칙을 코드로 옮긴 것이다.
- 입력은 설계 변수 네 개(H, e_ratio, N, U). e는 e_ratio x H라서 넣지 않는다.
- 출력은 dp, dT_max, dT_avg. 범위가 15배 넘는 dp는 로그로 바꿔 학습한다.
- 학습에는 valid가 True인 행만 쓰고, 나누기는 데이터셋의 split 열을 따른다.
- 표준화(평균 0, 표준편차 1)는 train으로만 맞춘다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = (Path(__file__).resolve().parent.parent / "data" / "cooling_v1"
        / "cases.parquet")
INPUTS = ["H", "e_ratio", "N", "U"]
TARGETS = ["dp", "dT_max", "dT_avg"]
LOG = np.array([t == "dp" for t in TARGETS])    # 로그로 바꿔 학습할 출력


def load(*splits):
    """split 이름(train, val, test, ood_U, ood_H)의 사용 가능한 행을 읽는다."""
    df = pd.read_parquet(DATA)
    df = df[df["valid"] & df["split"].isin(splits)].reset_index(drop=True)
    return df[INPUTS].to_numpy(float), df[TARGETS].to_numpy(float), df


def to_model(y):
    """출력을 학습용 값으로: dp만 자연로그."""
    z = np.array(y, dtype=float)
    z[:, LOG] = np.log(z[:, LOG])
    return z


def from_model(z):
    """학습용 값을 원래 단위의 출력으로 되돌린다."""
    y = np.array(z, dtype=float)
    y[:, LOG] = np.exp(y[:, LOG])
    return y


class Scaler:
    """열마다 평균을 빼고 표준편차로 나눈다. train으로만 fit 한다."""

    def fit(self, x):
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def rel_error(y_true, y_pred):
    """출력별 상대 오차 [%]의 평균과 최댓값을 dict로 돌려준다."""
    err = np.abs(y_pred / y_true - 1) * 100
    return {t: (err[:, i].mean(), err[:, i].max())
            for i, t in enumerate(TARGETS)}


def report(name, y_true, y_pred):
    """상대 오차를 한 줄로 출력한다."""
    e = rel_error(y_true, y_pred)
    print(f"{name:<22}" + "".join(f"{e[t][0]:7.2f}% {e[t][1]:6.1f}%"
                                   for t in TARGETS))
