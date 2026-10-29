"""책 07-4: FNO로 온도장·속도장·압력장을 학습한다 (GPU).

입력 (mask, U) 그림에서 출력 (Ux, Uy, p, dT) 그림을 예측한다.
손실은 케이스·채널마다 유체 격자점에서 구한 상대 L2 오차의 평균이다
(--loss mse면 정규화한 값의 평균 제곱 오차).

사용법: uv run python ch07/train_fno.py [옵션]
    --epochs 300 --modes 16 64 --width 32 --layers 4
    --step 2        한 칸씩 건너뛴 513 x 25 격자로 학습
    --no-flow       출력을 유속으로 무차원화하지 않음
    --out ~/cfd-runs/ch07/fno        (fno.pt와 fno.json 저장)
결과:   에포크 기록, 학습 시간, GPU 최대 메모리, val 상대 L2 오차
        RTX 3090에서 기본 설정 약 16분
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO

from fno_data import FieldScaler, load, split_tensors

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--modes", type=int, nargs=2, default=[16, 64])
p.add_argument("--width", type=int, default=32)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--padding", type=float, default=0.1)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--step", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no-flow", dest="flow", action="store_false",
               help="유속으로 무차원화하지 않고 RMS로만 나눈다")
p.add_argument("--loss", choices=["mse", "rel"], default="rel")
p.add_argument("--out", default="~/cfd-runs/ch07/fno")
args = p.parse_args()


def masked_mse(pred, y, mask):
    m = mask[:, None]
    return (((pred - y) * m) ** 2).sum() / (m.sum() * y.shape[1])


def rel_loss(pred, y, mask):
    """케이스·채널마다 상대 L2 오차를 구해 평균한다."""
    return rel_l2(pred, y, mask).mean()


def rel_l2(pred, y, mask):
    """케이스·채널마다 ||예측 - 정답|| / ||정답|| (유체 격자점만) -> (n, 4)."""
    m = mask[:, None]
    num = (((pred - y) * m) ** 2).sum(dim=(2, 3)).sqrt()
    return num / ((y * m) ** 2).sum(dim=(2, 3)).sqrt()


def predict(model, x, batch=16):
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch])
                          for i in range(0, len(x), batch)])


torch.manual_seed(args.seed)
data = load()
train = data["split"] == "train"
scaler = FieldScaler(args.flow).fit(data["fields"][train], data["mask"][train],
                                   data["inputs"][train, 3])
x_tr, y_tr, m_tr = split_tensors(data, "train", scaler, step=args.step)
x_va, y_va, m_va = split_tensors(data, "val", scaler, step=args.step)
print(f"train {tuple(x_tr.shape)} -> {tuple(y_tr.shape)}, val {len(x_va)}개")

model = FNO(n_modes=tuple(args.modes), in_channels=2, out_channels=4,
            hidden_channels=args.width, n_layers=args.layers,
            domain_padding=args.padding).cuda()
n_par = sum(p.numel() * (2 if p.is_complex() else 1)
            for p in model.parameters())
print(f"파라미터 {n_par / 1e6:.2f}M개 (실수 기준)")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

loss_fn = rel_loss if args.loss == "rel" else masked_mse

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
for epoch in range(1, args.epochs + 1):
    model.train()
    order = torch.randperm(len(x_tr), device="cuda")
    total = 0.0
    for i in range(0, len(order), args.batch):
        b = order[i:i + args.batch]
        loss = loss_fn(model(x_tr[b]), y_tr[b], m_tr[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
    sched.step()
    if epoch % 50 == 0 or epoch == 1:
        model.eval()
        va = rel_l2(predict(model, x_va), y_va, m_va).mean(0) * 100
        print(f"에포크 {epoch:4d}  train 손실 {total / len(x_tr):.2e}  "
              f"val 상대 L2 [%] " + " ".join(f"{v:.2f}" for v in va.tolist())
              + f"  ({time.perf_counter() - start:.0f}초)")
torch.cuda.synchronize()
sec = time.perf_counter() - start
print(f"학습 {sec:.0f}초 (에포크당 {sec / args.epochs:.2f}초), GPU 최대 메모리 "
      f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

out = Path(args.out).expanduser()
out.parent.mkdir(parents=True, exist_ok=True)
# neuraloperator는 state_dict에 생성 인자(_metadata)를 끼워 넣는데, 여기에
# 함수·클래스가 들어 있어 torch.load 기본값(가중치만 읽기)으로 열리지 않는다.
# 구조는 fno.json에 따로 적으므로 텐서만 저장한다
weights = {k: v for k, v in model.state_dict().items() if k != "_metadata"}
torch.save(weights, out.with_suffix(".pt"))
meta = dict(vars(args), rms=scaler.rms.tolist(), train_seconds=round(sec, 1),
            params=n_par)
out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print("저장:", out.with_suffix(".pt"))
