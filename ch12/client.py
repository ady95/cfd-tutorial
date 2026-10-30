"""책 12-3: API에 최적 설계를 묻고, 그 설계의 CFD 재검증까지 요청한다.

1) POST /optimize 로 대리모델 최적 설계를 받는다 (--design을 주면 건너뜀)
2) POST /verify 로 CFD 재검증 작업을 맡기고
3) GET /jobs/{작업} 을 10초마다 물어 끝나면 보고서를 출력한다

사용법: uv run python ch12/client.py [--p-max 1.0] [--design H e/H N U]
                                     [--url http://127.0.0.1:8000]
"""
import argparse
import json
import time
import urllib.request


def call(url, body=None):
    """JSON을 보내고 JSON을 받는다 (body가 없으면 GET)."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--p-max", type=float, default=1.0, help="동력 상한 비율")
ap.add_argument("--design", type=float, nargs=4, metavar=("H", "E", "N", "U"))
args = ap.parse_args()

if args.design:
    H, er, N, U = args.design
    design = {"H": H, "e_ratio": er, "N": int(N), "U": U}
else:
    start = time.time()
    r = call(f"{args.url}/optimize", {"p_max_ratio": args.p_max})
    design, p = r["design"], r["result"]
    print(f"최적 설계 ({time.time() - start:.1f}초): H {design['H']:.3f}"
          f" e/H {design['e_ratio']:.3f} N {design['N']} U {design['U']:.4f}")
    print(f"  예측 dT_max {p['dT_max']:.3f}K (기준 {r['base']['dT_max']:.3f}K)"
          f", P {p['P']:.3f} mW/m (상한 {r['p_max']:.3f})")

job = call(f"{args.url}/verify", {"design": design})["job"]
print(f"재검증 작업 {job} 접수")
start, last = time.time(), "queued"
while True:
    time.sleep(10)
    rec = call(f"{args.url}/jobs/{job}")
    if rec["status"] in ("done", "error"):
        break
    if rec["status"] != last:       # 상태가 바뀔 때만 출력한다
        print(f"  {time.time() - start:4.0f}초 {rec['status']}", flush=True)
        last = rec["status"]
print(f"작업 {rec['status']} ({time.time() - start:.0f}초)\n")
if rec["status"] == "done":
    print(rec["report"])
    print(f"feedback.csv에 더한 행: {rec['result'].get('feedback_added')}")
else:
    print(rec["error"])
