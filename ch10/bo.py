"""책 10-3, 10-5: BoTorch 베이지안 최적화로 제약이 있는 최소화 문제를 푼다.

evaluate(x) 하나만 바꾸면 대리모델(10-3)에도 CFD(10-5)에도 쓸 수 있다.
- evaluate: 설계 (n, 4) -> 목적 f (n,), 제약 c (n, k). c ≤ 0 이면 만족.
  계산이 실패한 설계는 f에 nan을 넣는다 (GP 학습에서 뺀다)
- 목적과 제약마다 가우스 과정(GP)을 하나씩 맞추고, 기대 개선량에 제약을
  만족할 확률을 곱한 값(의 로그, LogCEI)이 가장 큰 설계를 다음에 평가한다
- 탐색은 0~1 변수 공간에서 하고 problem.from_unit으로 설계로 바꾼다.
  틈 제약은 변환이 지켜 주고, N은 연속값으로 고른 뒤 반올림한다
"""
import warnings

import numpy as np
import torch
from scipy.stats import qmc

from problem import from_unit, to_unit

# BoTorch가 쓰는 linear_operator는 불러올 때 torch.jit 폐기 예고를 내고,
# GP 학습 중에도 수치 경고를 자주 낸다. 결과에 영향이 없으므로 숨긴다.
warnings.filterwarnings("ignore")
from botorch.acquisition.analytic import (  # noqa: E402
    LogConstrainedExpectedImprovement)
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import SingleTaskGP  # noqa: E402
from botorch.models.transforms import Standardize  # noqa: E402
from botorch.optim import optimize_acqf  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402

UNIT = torch.tensor([[0.0] * 4, [1.0] * 4], dtype=torch.double)


def next_design(x, f, c):
    """지금까지의 평가로 GP를 맞추고 다음에 평가할 설계 하나를 고른다."""
    ok = np.isfinite(f)
    X = torch.tensor(to_unit(x[ok]), dtype=torch.double)
    Y = torch.tensor(np.column_stack([f[ok], c[ok]]), dtype=torch.double)
    model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=Y.shape[1]))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    feas = (c[ok] <= 0).all(1)
    best = f[ok][feas].min() if feas.any() else f[ok].max()
    acq = LogConstrainedExpectedImprovement(
        model, best_f=best, objective_index=0, maximize=False,
        constraints={1 + j: (None, 0.0) for j in range(c.shape[1])})
    u, _ = optimize_acqf(acq, bounds=UNIT, q=1, num_restarts=8,
                         raw_samples=256)
    return from_unit(u.detach().numpy())


def bo_minimize(evaluate, budget, seed=0, n_init=10, verbose=False):
    """평가 budget회로 최소화한다 -> 평가 순서대로 (f, c, x)."""
    torch.manual_seed(seed)
    x = from_unit(qmc.Sobol(4, seed=seed).random(n_init))
    f, c = evaluate(x)
    while len(f) < budget:
        x_new = next_design(x, f, c)
        f_new, c_new = evaluate(x_new)
        x, f, c = np.vstack([x, x_new]), np.r_[f, f_new], np.vstack([c, c_new])
        if verbose:
            feas = np.isfinite(f) & (c <= 0).all(1)
            print(f"  평가 {len(f):3d}회  지금까지 최소 "
                  f"{f[feas].min() if feas.any() else np.nan:.4f}", flush=True)
    return f, c, x
