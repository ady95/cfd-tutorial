"""책 05-5: Optuna로 XGBoost와 MLP의 하이퍼파라미터를 찾고 최종 모델을 저장한다.

1) XGBoost: 트리 깊이·학습률 등 다섯 개를 50번 시도한다.
2) MLP: 너비·깊이·학습률·활성화 함수를 30번 시도한다. 탐색 중에는 5000
   에포크만 학습하고, 성적이 나쁜 시도는 중간에 멈춘다(가지치기).
3) MLP 최고 설정을 20000 에포크로 다시 학습해 05-4의 기본 설정과 비교하고,
   val 오차가 더 작은 쪽을 최종 모델로 저장한다.
목표값은 val에서 세 출력의 평균 상대 오차 [%]의 평균이다.

사용법: uv run python ch05/tune.py [출력 폴더]   (기본 ~/cfd-runs/ch05)
결과:   표 출력, 출력 폴더의 tuning.png, 저장소의 models/surrogate_v1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import optuna
import torch
import xgboost as xgb
from torch import nn

from data import (DATA, INPUTS, TARGETS, Scaler, from_model, load,
                  rel_error, to_model)

matplotlib.use("Agg")       # 화면 없이 PNG로만 저장
import matplotlib.pyplot as plt  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
ACTS = {"SiLU": nn.SiLU, "Tanh": nn.Tanh}
SEARCH_EPOCHS, FINAL_EPOCHS = 5000, 20000

x_tr, y_tr, _ = load("train")
x_va, y_va, _ = load("val")
z_tr, z_va = to_model(y_tr), to_model(y_va)


def score(y_pred):
    """val에서 세 출력 평균 상대 오차의 평균 [%]."""
    return float(np.mean([e[0] for e in rel_error(y_va, y_pred).values()]))


# 1) XGBoost: 출력마다 학습하고 val로 조기 종료
def xgb_objective(trial):
    p = {"objective": "reg:squarederror", "seed": 0,
         "max_depth": trial.suggest_int("max_depth", 2, 8),
         "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
         "min_child_weight": trial.suggest_float("min_child_weight", 1, 20,
                                                 log=True),
         "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True)}
    pred = []
    for i in range(len(TARGETS)):
        d_tr = xgb.DMatrix(x_tr, label=z_tr[:, i])
        d_va = xgb.DMatrix(x_va, label=z_va[:, i])
        b = xgb.train(p, d_tr, 5000, evals=[(d_va, "val")],
                      early_stopping_rounds=100, verbose_eval=False)
        pred.append(b.predict(d_va, iteration_range=(0, b.best_iteration + 1)))
    return score(from_model(np.column_stack(pred)))


# 2) MLP: 05-4와 같은 학습 방식, 500 에포크마다 val을 보고 가지치기
sx, sz = Scaler().fit(x_tr), Scaler().fit(z_tr)
X = torch.tensor(sx.transform(x_tr), dtype=torch.float32)
Z = torch.tensor(sz.transform(z_tr), dtype=torch.float32)
XV = torch.tensor(sx.transform(x_va), dtype=torch.float32)


def build(act, width, depth):
    layers, n = [], len(INPUTS)
    for _ in range(depth):
        layers += [nn.Linear(n, width), ACTS[act]()]
        n = width
    return nn.Sequential(*layers, nn.Linear(n, len(TARGETS)))


def predict(model, x):
    with torch.no_grad():
        z = model(x).numpy().astype(float)
    return from_model(sz.inverse(z))


def train_mlp(cfg, epochs, trial=None):
    torch.manual_seed(0)
    model = build(cfg["act"], cfg["width"], cfg["depth"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        ((model(X) - Z) ** 2).mean().backward()
        opt.step()
        sched.step()
        if trial is not None and ep % 500 == 0:
            trial.report(score(predict(model, XV)), ep)
            if trial.should_prune():        # 같은 시점의 다른 시도보다 나쁘면
                raise optuna.TrialPruned()  # 여기서 멈춘다
    return model


def mlp_objective(trial):
    cfg = {"act": trial.suggest_categorical("act", list(ACTS)),
           "width": trial.suggest_categorical("width", [32, 64, 128]),
           "depth": trial.suggest_int("depth", 2, 4),
           "lr": trial.suggest_float("lr", 3e-4, 1e-2, log=True)}
    return score(predict(train_mlp(cfg, SEARCH_EPOCHS, trial), XV))


studies = {}
for name, objective, n_trials, pruner in [
        ("XGBoost", xgb_objective, 50, optuna.pruners.NopPruner()),
        ("MLP", mlp_objective, 30,
         optuna.pruners.MedianPruner(n_startup_trials=5,
                                     n_warmup_steps=1000))]:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0),
                                pruner=pruner)
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    pruned = sum(t.state.name == "PRUNED" for t in study.trials)
    print(f"[{name}] {n_trials}회 {time.perf_counter() - start:5.0f}초,"
          f" 가지치기 {pruned}회, 최고 {study.best_value:.3f}%")
    print("  " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else
                           f"{k}={v}" for k, v in study.best_params.items()))
    studies[name] = study

# 3) MLP 최고 설정을 20000 에포크로 다시 학습해 05-4 기본 설정과 비교
default = {"act": "SiLU", "width": 64, "depth": 3, "lr": 1e-3}
best = studies["MLP"].best_params
print(f"\nMLP {FINAL_EPOCHS} 에포크        dp      dT_max  dT_avg   평균")
final = {}
for name, cfg in [("05-4 기본 설정", default), ("Optuna 최고 설정", best)]:
    model = train_mlp(cfg, FINAL_EPOCHS)
    e = rel_error(y_va, predict(model, XV))
    final[name] = (score(predict(model, XV)), cfg, model, e)
    pad = 18 - sum(2 if ord(c) > 0x1100 else 1 for c in name)
    print(name + " " * pad + "".join(f"{e[t][0]:7.3f}%" for t in TARGETS)
          + f" {final[name][0]:6.3f}%")

name = min(final, key=lambda k: final[k][0])
_, cfg, model, e = final[name]
out_model = Path(__file__).resolve().parent.parent / "models" / "surrogate_v1"
out_model.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out_model / "mlp.pt")
meta = {"model": "MLP", "chosen": name, "config": cfg,
        "epochs": FINAL_EPOCHS, "inputs": INPUTS, "targets": TARGETS,
        "log_targets": ["dp"],
        "x_scaler": {"mean": sx.mean.tolist(), "std": sx.std.tolist()},
        "z_scaler": {"mean": sz.mean.tolist(), "std": sz.std.tolist()},
        "val_mean_rel_error_pct": {t: round(e[t][0], 4) for t in TARGETS},
        "dataset": DATA.parent.name,
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest()}
(out_model / "mlp.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
print(f"\n최종 모델: {name} -> {out_model}")

# 탐색 기록 그림: 시도할수록 최고 기록이 어떻게 줄었는지
out = Path(sys.argv[1] if len(sys.argv) > 1 else "~/cfd-runs/ch05")
out = out.expanduser()
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, study) in zip(axes, studies.items()):
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    ax.scatter([t.number for t in done], [t.value for t in done], s=20,
               color="0.6", label="completed trial")
    ax.step(range(len(study.trials)), np.minimum.accumulate(
        [t.value if t.value is not None else np.inf for t in study.trials]),
        where="post", color="#2a78d6", lw=2, label="best so far")
    pruned = len(study.trials) - len(done)
    ax.set(yscale="log", xlabel="trial", ylabel="val mean relative error [%]",
           title=f"({'ab'[list(studies).index(name)]}) {name}"
           + (f" ({pruned} pruned trials not shown)" if pruned else ""))
    for axis in (ax.yaxis.set_major_formatter, ax.yaxis.set_minor_formatter):
        axis(FormatStrFormatter("%g"))
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(out / "tuning.png", dpi=150)
