"""책 06-4: CFD와 대리모델의 계산 시간을 재고, 총비용과 손익분기점을 계산한다.

1) 측정 환경(CPU, GPU, 라이브러리 버전, 스레드 수)을 출력한다.
2) CFD 한 건의 시간은 4장 데이터셋에 기록된 해석 시간으로 구한다.
3) 최종 MLP와 3차 다항 회귀의 추론 시간을 한 건씩, 그리고 10만 건 묶음으로 잰다.
   MLP 묶음 추론은 GPU로도 잰다.
4) 데이터 생성·학습 비용을 더한 총비용으로 손익분기점을 계산한다.

사용법: uv run python ch06/speed.py   (약 1분. 다른 계산이 없는 상태에서 실행)
"""
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import torch

from common import ROOT, fit_poly, load, load_mlp, load_mlp_parts

# 4장·5장에서 잰 시간 [초]
BATCH_WALL = 24.6 * 60      # 04-3: 700개를 동시 6개로 해석한 실제 시간
RERUN_WALL = 733            # 04-4: 늦게 수렴한 3개 재해석 (가장 긴 케이스)
TRAIN_MLP = 31.5            # 05-4: MLP 20000 에포크 학습
N_CASES = 700


def timeit(f, repeat):
    """f를 repeat번 실행한 한 번당 평균 시간 [초]. 처음 몇 번은 버린다."""
    for _ in range(5):
        f()
    start = time.perf_counter()
    for _ in range(repeat):
        f()
    return (time.perf_counter() - start) / repeat


# 1) 측정 환경
cpu = next((ln.split(":")[1].strip() for ln in open("/proc/cpuinfo")
            if ln.startswith("model name")), platform.processor())
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음"
print(f"CPU {cpu} | GPU {gpu}")
print(f"Python {platform.python_version()} | PyTorch {torch.__version__} | "
      f"scikit-learn {sklearn.__version__} | 스레드 {torch.get_num_threads()}")

# 2) CFD 한 건: 데이터셋에 기록된 케이스별 해석 시간 (코어 하나)
df = pd.read_parquet(ROOT / "data" / "cooling_v1" / "cases.parquet")
sec = df["seconds"]
cfd_case = sec.median()
cfd_batch = BATCH_WALL / N_CASES        # 동시 6개일 때 한 건당 실제 시간
print(f"\nCFD 해석 시간(코어 하나): 중앙값 {sec.median():.1f}초,"
      f" 평균 {sec.mean():.1f}초, 최대 {sec.max():.0f}초,"
      f" 합계 {sec.sum() / 3600:.1f}시간")
print(f"CFD 동시 6개: {N_CASES}개 {BATCH_WALL / 60:.1f}분 -> "
      f"한 건당 {cfd_batch:.2f}초")

# 3) 추론 시간
x_tr, y_tr, _ = load("train")
mlp, poly = load_mlp(), fit_poly(x_tr, y_tr, 3)
one = x_tr[:1]
rng = np.random.default_rng(0)
many = x_tr[rng.integers(0, len(x_tr), 100_000)]
rows = [("MLP CPU, 1건씩", timeit(lambda: mlp(one), 2000), 1),
        ("MLP CPU, 10만 건 묶음", timeit(lambda: mlp(many), 5), len(many)),
        ("3차 다항, 1건씩", timeit(lambda: poly(one), 2000), 1),
        ("3차 다항, 10만 건 묶음", timeit(lambda: poly(many), 5), len(many))]
if torch.cuda.is_available():
    model, sx, sz = load_mlp_parts()
    model = model.cuda()

    def mlp_gpu():
        xs = torch.tensor(sx.transform(many), dtype=torch.float32,
                          device="cuda")
        with torch.no_grad():
            z = model(xs).cpu().numpy()         # 결과를 CPU로 가져올 때까지
        return sz.inverse(z)
    rows.append(("MLP GPU, 10만 건 묶음", timeit(mlp_gpu, 5), len(many)))


def fmt(t):
    """초 단위 시간을 읽기 좋은 단위로."""
    if t >= 1:
        return f"{t:.1f} s"
    return f"{t * 1e3:.2f} ms" if t >= 1e-3 else f"{t * 1e6:.2f} µs"


def width(s):
    return sum(2 if ord(c) > 0x1100 else 1 for c in s)   # 한글은 2칸


print("\n방법" + " " * 22 + "호출 한 번      한 건당    CFD 1건 대비")
print("CFD 1건(중앙값)" + " " * 9 + f"{fmt(cfd_case):>10}{fmt(cfd_case):>13}"
      + f"{'1배':>14}")
for name, t, n in rows:
    print(name + " " * (24 - width(name)) + f"{fmt(t):>10}"
          f"{fmt(t / n):>13}" + f"{cfd_case / (t / n):>14,.0f}배")

# 4) 총비용과 손익분기점: 동시 6개로 해석하는 경우와 비교
cost = BATCH_WALL + RERUN_WALL + TRAIN_MLP
t_inf = rows[0][1]
print(f"\n대리모델 준비 비용: 데이터 생성 {BATCH_WALL / 60:.1f}분 + 재해석 "
      f"{RERUN_WALL / 60:.1f}분 + 학습 {TRAIN_MLP:.0f}초 = {cost / 60:.1f}분")
print(f"손익분기점: {cost:.0f}초 / ({cfd_batch:.2f}초 - {t_inf * 1e3:.2f}ms)"
      f" = 약 {cost / (cfd_batch - t_inf):.0f}건")
for n in (100, 1000, 10_000, 100_000):
    print(f"  {n:>7,}건 평가: CFD {n * cfd_batch / 3600:7.1f}시간, "
          f"대리모델 {(cost + n * t_inf) / 3600:5.2f}시간")
