"""책 07-2: FNO의 파라미터 수, 학습 한 스텝 시간, GPU 메모리를 잰다.

해상도, 모드 수, 채널 폭을 바꿔 가며 묶음 8개로 순전파·역전파를 한 번 하는
시간과 그때의 GPU 최대 메모리를 잰다. 입력은 난수라 학습은 하지 않는다.

사용법: uv run python ch07/fno_cost.py      (GPU가 비어 있을 때 실행, 약 30초)
"""
import time

import torch
from neuralop.models import FNO

CASES = [  # (ny, nx, 모드 (y, x), 채널 폭)
    (25, 513, (16, 64), 32),
    (49, 1025, (16, 64), 32),
    (97, 2049, (16, 64), 32),
    (49, 1025, (8, 32), 32),
    (49, 1025, (24, 128), 32),
    (49, 1025, (16, 64), 64),
]


def count(model):
    """실수 개수로 센다. 복소수 가중치 하나는 실수 2개."""
    spec = other = 0
    for name, p in model.named_parameters():
        n = p.numel() * (2 if p.is_complex() else 1)
        if "convs" in name and "weight" in name:
            spec += n
        else:
            other += n
    return spec, other


print("격자          모드      폭   푸리에 가중치   나머지   스텝 [ms]  메모리 [GB]")
for ny, nx, modes, width in CASES:
    torch.manual_seed(0)
    model = FNO(n_modes=modes, in_channels=2, out_channels=4,
                hidden_channels=width, domain_padding=0.1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 2, ny, nx, device="cuda")
    y = torch.randn(8, 4, ny, nx, device="cuda")

    def step():
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(3):                  # 준비 실행은 버린다
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(20):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / 20 * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9
    spec, other = count(model)
    print(f"{ny:3d} x {nx:4d}  {modes[0]:2d} x {modes[1]:3d}  {width:3d}"
          f"  {spec / 1e6:10.2f}M  {other / 1e6:7.2f}M  {ms:9.1f}  {mem:10.2f}")
    del model, opt, x, y
    torch.cuda.empty_cache()
