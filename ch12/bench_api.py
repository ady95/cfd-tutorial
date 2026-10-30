"""책 12-2: 예측 API의 응답 시간을 잰다.

1) 같은 프로세스에서 모델을 직접 부를 때 (API를 거치지 않은 기준)
2) /predict 설계 1개 요청을 차례로 300번: 클라이언트가 잰 왕복 시간
3) /predict 한 요청에 설계 100개, 1만 개, 10만 개
4) 동시 요청: 스레드 8개가 설계 1개씩 800번
5) /optimize 3번

사용법: API를 띄운 뒤 uv run python ch12/bench_api.py [--url ...]
결과:   표 출력과 작업 폴더의 bench.json (12-4의 cost.py가 읽는다)
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import system


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def designs(n, seed=0):
    """학습 범위 안의 무작위 설계 n개 (요청 본문 형식)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform([1.5, 0.2, 4, 0.05], [2.5, 0.4, 16, 0.15], (n, 4))
    return [{"H": a, "e_ratio": b, "N": int(c), "U": d} for a, b, c, d in x]


def timed(f):
    start = time.perf_counter()
    out = f()
    return (time.perf_counter() - start) * 1000, out


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
args = ap.parse_args()
info = json.loads(urllib.request.urlopen(f"{args.url}/info").read())
print(f"모델: {info['model']} ({info['dir']})")

# 1) API 없이 직접
sur = system.Surrogate(info["dir"])
one = [[2.0, 0.3, 8, 0.1]]
for _ in range(20):
    sur.describe(one)
t = [timed(lambda: sur.describe(one))[0] for _ in range(300)]
rec = {"direct_ms": float(np.median(t))}
print(f"\n직접 호출, 설계 1개      중앙값 {np.median(t):7.2f} ms")

# 2) API 설계 1개, 차례로
url = f"{args.url}/predict"
body = {"designs": designs(1)}
for _ in range(20):
    post(url, body)
res = [timed(lambda: post(url, body)) for _ in range(300)]
t = np.array([a for a, _ in res])
srv = np.array([r["ms"] for _, r in res])
rec["predict_ms"] = float(np.median(t))
print(f"API 설계 1개 (300번)     중앙값 {np.median(t):7.2f} ms,"
      f" 95% {np.percentile(t, 95):6.2f} ms (서버 안 계산 {np.median(srv):.2f})")

# 3) 한 요청에 여러 설계
for n in (100, 10_000, 100_000):
    body = {"designs": designs(n, seed=n)}
    t, r = min((timed(lambda: post(url, body)) for _ in range(3)),
               key=lambda a: a[0])
    rec[f"batch_{n}_ms"] = t
    print(f"API 설계 {n:>7,}개 한 번에 {t:9.1f} ms (서버 안 {r['ms']:8.1f}),"
          f" 설계당 {t / n * 1000:7.1f} µs")

# 4) 동시 요청
body = {"designs": designs(1)}
start = time.perf_counter()
with ThreadPoolExecutor(8) as pool:
    t = list(pool.map(lambda _: timed(lambda: post(url, body))[0],
                      range(800)))
sec = time.perf_counter() - start
rec["concurrent_rps"] = 800 / sec
print(f"동시 8개, 800번          {800 / sec:7.0f} 요청/초,"
      f" 응답 중앙값 {np.median(t):.1f} ms")

# 5) 최적화
t = [timed(lambda: post(f"{args.url}/optimize", {}))[0] / 1000
     for _ in range(3)]
print(f"/optimize 3번            {np.mean(t):7.1f} 초 (각 "
      + ", ".join(f"{a:.1f}" for a in t) + ")")
rec["optimize_s"] = float(np.mean(t))
out = system.load_config(system.ROOT / "ch12" / "config.toml")["out"]
(out / "bench.json").write_text(json.dumps(rec, indent=2) + "\n")
