"""책 08-3: PhysicsNeMo의 FNO를 7장과 같은 데이터·같은 조건으로 학습한다 (GPU).

입력·출력·정규화·손실·옵티마이저·에포크는 ch07/train_fno.py와 같다.
모델만 physicsnemo.models.fno.FNO로 바꾼다. 7장 설정과 맞추려면
- 모드: neuraloperator n_modes=(16, 64)는 y 방향 16개(양·음 8개씩),
  x 방향 64 // 2 + 1 = 33개를 남긴다. PhysicsNeMo는 첫 번째 값을
  양·음 양쪽에 쓰므로 [8, 33]이 같은 모드 수다
- 패딩: neuraloperator는 양 끝에 10%씩(49 x 1025 -> 59 x 1229),
  PhysicsNeMo는 한쪽 끝에만 덧댄다. 늘어난 크기를 맞춰 [10, 204]

사용법: uv run python ch08/train_pnfno.py [옵션]
    --epochs 300 --modes 8 33 --padding 10 204 --width 32 --layers 4
    --decoder-layers 1 --decoder-size 32
    --step 2        한 칸씩 건너뛴 513 x 25 격자로 학습 (패딩은 절반으로)
    --out ~/cfd-runs/ch08/pnfno      (pnfno.mdlus와 pnfno.json 저장)
결과:   에포크 기록, 학습 시간, GPU 최대 메모리
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from physicsnemo.models.fno import FNO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch07"))
from fno_data import FieldScaler, load, split_tensors  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--modes", type=int, nargs=2, default=[8, 33])
p.add_argument("--padding", type=int, nargs=2, default=[10, 204])
p.add_argument("--width", type=int, default=32)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--decoder-layers", type=int, default=1)
p.add_argument("--decoder-size", type=int, default=32)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--step", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="~/cfd-runs/ch08/pnfno")
args = p.parse_args()


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
scaler = FieldScaler().fit(data["fields"][train], data["mask"][train],
                           data["inputs"][train, 3])
x_tr, y_tr, m_tr = split_tensors(data, "train", scaler, step=args.step)
x_va, y_va, m_va = split_tensors(data, "val", scaler, step=args.step)
print(f"train {tuple(x_tr.shape)} -> {tuple(y_tr.shape)}, val {len(x_va)}개")

model = FNO(in_channels=2, out_channels=4, dimension=2,
            latent_channels=args.width, num_fno_layers=args.layers,
            num_fno_modes=args.modes,
            padding=[v // args.step for v in args.padding],
            decoder_layers=args.decoder_layers,
            decoder_layer_size=args.decoder_size).cuda()
n_par = sum(p.numel() for p in model.parameters())
print(f"파라미터 {n_par / 1e6:.2f}M개")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
for epoch in range(1, args.epochs + 1):
    model.train()
    order = torch.randperm(len(x_tr), device="cuda")
    total = 0.0
    for i in range(0, len(order), args.batch):
        b = order[i:i + args.batch]
        loss = rel_l2(model(x_tr[b]), y_tr[b], m_tr[b]).mean()
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
# .mdlus 파일에 생성 인자와 가중치가 함께 들어간다.
# 정규화 값은 모델 밖의 일이므로 json에 따로 적는다
model.save(str(out.with_suffix(".mdlus")))
meta = dict(vars(args), rms=scaler.rms.tolist(), train_seconds=round(sec, 1),
            params=n_par)
out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print("저장:", out.with_suffix(".mdlus"))
