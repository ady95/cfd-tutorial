"""책 09-3: 데이터 손실에 질량·에너지 보존 손실을 더해 FNO를 학습한다 (GPU).

모델, 데이터, 정규화, 옵티마이저는 ch07/train_fno.py와 같다. 손실만 다르다.
    손실 = 데이터 손실 (상대 L2, 07-4)
         + λ_m x 질량 손실 + λ_e x 에너지 손실
질량·에너지 손실은 케이스마다 단면 잔차(physics.py)의 RMS를 구해 평균한 값이다.
상대 L2처럼 "비율"이라 λ = 1이면 두 손실이 비슷한 크기가 된다.

물리 손실은 정답(CFD)이 없어도 계산할 수 있다. --colloc N을 주면 설계 변수를
무작위로 N개 더 뽑아, 이 입력에는 물리 손실만 건다.

사용법: uv run python ch09/train_pifno.py [옵션]
    --lam-mass 1 --lam-energy 1   물리 손실 가중치 (0이면 7장과 같음)
    --warmup 50                   처음 몇 에포크 동안 λ를 0에서 선형으로 올림
    --n-train 100                 train에서 앞의 100개만 사용 (고정된 무작위 순서)
    --colloc 400 --colloc-range wide   정답 없는 입력 400개 (in 또는 wide)
    --step 2                      한 칸씩 건너뛴 513 x 25 격자로 학습
    --out ~/cfd-runs/ch09/pifno   (pifno.pt와 pifno.json, 7장과 같은 형식)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO

from physics import all_weights, expected, residuals, weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch07"))
from fno_data import (U_REF, FieldScaler, load, make_inputs,  # noqa: E402
                      split_tensors)
from grid import fluid_mask  # noqa: E402

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
p.add_argument("--lam-mass", type=float, default=1.0)
p.add_argument("--lam-energy", type=float, default=1.0)
p.add_argument("--warmup", type=int, default=0)
p.add_argument("--tol", type=float, nargs=2, default=[0, 0],
               help="질량·에너지 잔차의 허용 오차. 이보다 작은 잔차는 벌점 없음")
p.add_argument("--n-train", type=int, default=0, help="0이면 train 전체")
p.add_argument("--colloc", type=int, default=0)
p.add_argument("--colloc-range", choices=["in", "wide"], default="in")
p.add_argument("--out", default="~/cfd-runs/ch09/pifno")
args = p.parse_args()
# 확장자를 붙일 때 with_suffix를 쓰므로 이름에 점이 있으면 뒷부분이 잘린다
assert "." not in Path(args.out).name, "--out 이름에 점(.)을 쓰지 마십시오"

RANGES = {"in": {"H": (1.5, 2.5), "U": (0.05, 0.15)},      # 4장 설계 범위
          "wide": {"H": (1.5, 3.0), "U": (0.05, 0.20)}}     # ood_H, ood_U까지


def rel_l2(pred, y, mask):
    """케이스·채널마다 ||예측 - 정답|| / ||정답|| (유체 격자점만) -> (n, 4)."""
    m = mask[:, None]
    num = (((pred - y) * m) ** 2).sum(dim=(2, 3)).sqrt()
    return num / ((y * m) ** 2).sum(dim=(2, 3)).sqrt()


def ramp_weights(epoch, warmup, lam_m, lam_e):
    """처음 warmup 에포크 동안 λ를 0에서 선형으로 올린다 (epoch는 1부터)."""
    ramp = min(1.0, epoch / warmup) if warmup else 1.0
    return lam_m * ramp, lam_e * ramp


def soft_threshold(r, tol):
    """허용 오차 안의 잔차는 0으로, 밖이면 넘은 만큼만 남긴다 (09-5)."""
    return torch.sign(r) * (r.abs() - tol).clamp(min=0)


def physics_loss(z, u, W, q, e, tol=(0, 0)):
    """정규화한 예측 z와 U/U_REF -> (질량 손실, 에너지 손실) 케이스 평균."""
    Ux = z[:, 0] * rms[0] * u[:, None, None]   # FieldScaler.inverse와 같음
    dT = z[:, 3] * rms[3]
    rm, re = residuals(Ux, dT, W, q, e)
    rm, re = soft_threshold(rm, tol[0]), soft_threshold(re, tol[1])
    return rm.pow(2).mean(1).sqrt().mean(), re.pow(2).mean(1).sqrt().mean()


def predict(model, x, batch=16):
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch])
                          for i in range(0, len(x), batch)])


def physics_tensors(data, k, step):
    """split 마스크 k의 W, 기대 유량·열 유량, U/U_REF (GPU)."""
    sub = {key: data[key][k] for key in ("mask", "inputs")}
    sub["mask"] = sub["mask"][:, ::step, ::step]
    W, q, e = all_weights(sub)
    u = sub["inputs"][:, 3] / U_REF
    return [torch.as_tensor(a, device="cuda") for a in (W, q, e, u)]


def colloc_inputs(n, rng, step):
    """정답 없는 설계 n개: 입력 텐서와 물리 텐서."""
    r = RANGES[args.colloc_range]
    H = rng.uniform(*r["H"], n)
    er = rng.uniform(0.2, 0.4, n)
    N = rng.integers(4, 16, n)
    U = rng.uniform(*r["U"], n)
    ny, nx = (49 - 1) // step + 1, (1025 - 1) // step + 1
    mask = np.stack([fluid_mask(h, h * e_, k, nx, ny)
                     for h, e_, k in zip(H, er, N)])
    W = np.stack([weights(m, h, h * e_, k)
                  for m, h, e_, k in zip(mask, H, er, N)])
    q, e = map(np.stack, zip(*(expected(h, u_, nx) for h, u_ in zip(H, U))))
    x = make_inputs(mask, U)
    as_t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device="cuda")
    return as_t(x), [as_t(a) for a in (W, q, e, U / U_REF)]


torch.manual_seed(args.seed)
data = load()
train = data["split"] == "train"
if args.n_train:
    # 시드와 관계없이 같은 부분집합: 고정된 순서의 앞 n개
    idx = np.flatnonzero(train)[np.random.default_rng(0).permutation(
        train.sum())[:args.n_train]]
    train = np.zeros_like(train)
    train[idx] = True
    data["split"] = np.where(train, "train",
                             np.where(data["split"] == "train", "unused",
                                      data["split"]))
scaler = FieldScaler().fit(data["fields"][train], data["mask"][train],
                           data["inputs"][train, 3])
rms = torch.as_tensor(scaler.rms, device="cuda")
x_tr, y_tr, m_tr = split_tensors(data, "train", scaler, step=args.step)
x_va, y_va, m_va = split_tensors(data, "val", scaler, step=args.step)
P_tr = physics_tensors(data, train, args.step)
P_va = physics_tensors(data, data["split"] == "val", args.step)
if args.colloc:
    x_co, P_co = colloc_inputs(args.colloc, np.random.default_rng(1), args.step)
print(f"train {len(x_tr)}개, val {len(x_va)}개, 정답 없는 입력 {args.colloc}개 "
      f"({args.colloc_range if args.colloc else '-'}), "
      f"λ 질량 {args.lam_mass}, 에너지 {args.lam_energy}")

model = FNO(n_modes=tuple(args.modes), in_channels=2, out_channels=4,
            hidden_channels=args.width, n_layers=args.layers,
            domain_padding=args.padding).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
for epoch in range(1, args.epochs + 1):
    lm, le = ramp_weights(epoch, args.warmup, args.lam_mass, args.lam_energy)
    model.train()
    order = torch.randperm(len(x_tr), device="cuda")
    total = np.zeros(3)
    for i in range(0, len(order), args.batch):
        b = order[i:i + args.batch]
        z = model(x_tr[b])
        l_data = rel_l2(z, y_tr[b], m_tr[b]).mean()
        l_m, l_e = physics_loss(z, *[t[b] for t in
                                     (P_tr[3], P_tr[0], P_tr[1], P_tr[2])],
                                args.tol)
        loss = l_data + lm * l_m + le * l_e
        if args.colloc and (lm or le):
            c = torch.randint(len(x_co), (args.batch,), device="cuda")
            zc = model(x_co[c])
            c_m, c_e = physics_loss(zc, P_co[3][c], P_co[0][c], P_co[1][c],
                                    P_co[2][c], args.tol)
            loss = loss + lm * c_m + le * c_e
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += np.array([l_data.item(), l_m.item(), l_e.item()]) * len(b)
    sched.step()
    if epoch % 50 == 0 or epoch == 1:
        model.eval()
        z = predict(model, x_va)
        va = rel_l2(z, y_va, m_va).mean(0) * 100
        with torch.no_grad():
            vm, ve = physics_loss(z, P_va[3], P_va[0], P_va[1], P_va[2])
        t = total / len(x_tr)
        sec = time.perf_counter() - start
        print(f"에포크 {epoch:4d}  train 손실: 데이터 {t[0]:.4f}  "
              f"질량 {t[1]:.4f}  에너지 {t[2]:.4f}  ({sec:.0f}초)")
        l2 = " ".join(f"{v:5.2f}" for v in va.tolist())
        print(f"            val L2 [%] {l2}   보존 [%] 질량 "
              f"{vm.item() * 100:.2f} 에너지 {ve.item() * 100:.2f}")
torch.cuda.synchronize()
sec = time.perf_counter() - start
print(f"학습 {sec:.0f}초 (에포크당 {sec / args.epochs:.2f}초), GPU 최대 메모리 "
      f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

out = Path(args.out).expanduser()
out.parent.mkdir(parents=True, exist_ok=True)
weights_ = {k: v for k, v in model.state_dict().items() if k != "_metadata"}
torch.save(weights_, out.with_suffix(".pt"))
meta = dict(vars(args), flow=True, rms=scaler.rms.tolist(),
            train_seconds=round(sec, 1))
out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print("저장:", out.with_suffix(".pt"))
