"""책 부록 A-5: MeshGraphNet으로 냉각 채널의 셀 값 물리장을 학습한다.

PhysicsNeMo의 MeshGraphNet 기본 설정(처리 층 15개, 폭 128)을 쓴다.
그래프 2개를 한 묶음으로 학습하고, 학습률은 코사인으로 1e-3 -> 1e-5.

사용법 (PhysicsNeMo 컨테이너 안):
    python /tutorial/appendix/a5_mgn/train_mgn.py /out/graphs.pt /out [에포크]
    python ... --check       정답 셀 값으로 세 값을 다시 계산해 데이터셋과 비교
결과:   /out/mgn.pt (모델과 RMS), /out/train.log
"""
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from physicsnemo.models.meshgraphnet import MeshGraphNet

from mgn_common import batch, fit_rms, scalars, to_data

SEED = 0
BATCH = 2


def check(graphs):
    """정답 셀 값으로 다시 계산한 세 값과 데이터셋 값의 상대 차이 [%]."""
    diff = np.array([np.abs(scalars(g["y"], g) / np.array(g["cfd"]) - 1)
                     for g in graphs]) * 100
    print("셀 값으로 다시 계산한 세 값 - 데이터셋 값 (평균/최대 %)")
    for k, name in enumerate(["dp", "dT_max", "dT_avg"]):
        print(f"  {name:<7} {diff[:, k].mean():.3f} / {diff[:, k].max():.3f}")


def main():
    graphs = torch.load(sys.argv[1], weights_only=False)
    if "--check" in sys.argv:
        check(graphs)
        return
    out = Path(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    torch.manual_seed(SEED)
    random.seed(SEED)
    dev = "cuda"

    train = [g for g in graphs if g["split"] == "train"]
    val = [g for g in graphs if g["split"] == "val"]
    rms = fit_rms(train)
    tr = [to_data(g, rms, dev) for g in train]
    va = [to_data(g, rms, dev) for g in val]

    model = MeshGraphNet(input_dim_nodes=tr[0].x.shape[1],
                         input_dim_edges=tr[0].edge_attr.shape[1],
                         output_dim=4).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = epochs * ((len(tr) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, 1e-5)
    log = open(out / "train.log", "w")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"train {len(tr)}개, val {len(va)}개, 파라미터 {n_par:,}개, "
        f"에포크 {epochs}, 묶음 {BATCH}")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        order = list(range(len(tr)))
        random.shuffle(order)
        total = 0.0
        for i in range(0, len(order), BATCH):
            b = batch([tr[j] for j in order[i:i + BATCH]])
            loss = torch.nn.functional.mse_loss(
                model(b.x, b.edge_attr, b), b.y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                vl = np.mean([torch.nn.functional.mse_loss(
                    model(d.x, d.edge_attr, d), d.y).item() for d in va])
            say(f"에포크 {ep:4d}  train {total / len(tr):.5f}  val {vl:.5f}  "
                f"{time.perf_counter() - start:6.0f}초")
    sec = time.perf_counter() - start
    say(f"학습 {sec:.0f}초, GPU 최대 메모리 "
        f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    torch.save({"model": model.state_dict(), "rms": rms,
                "dims": (tr[0].x.shape[1], tr[0].edge_attr.shape[1]),
                "seconds": sec}, out / "mgn.pt")


if __name__ == "__main__":
    main()
