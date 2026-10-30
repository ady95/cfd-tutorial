"""책 부록 A-7: DrivAerML에서 대리모델에 필요한 작은 파일만 내려받는다.

DrivAerML (Ashton 외 2024, CC-BY-SA-4.0)은 DrivAer 노치백 차량을 500가지로
변형해 OpenFOAM으로 고해상도 해석한 공개 데이터셋이다. 전체는 약 31TB지만,
형상 변수와 힘 계수만 쓰면 CSV 세 개(약 150KB)로 충분하다.

내려받는 것
- geo_parameters_all.csv  형상 변수 16개 (500행)
- force_mom_all.csv       시간 평균 힘 계수 cd, cl, ... (공개된 484행)
- splits/manifest.json    데이터셋이 권장하는 학습·검증·시험 분할
그리고 run_1 폴더의 파일 크기 목록 (내려받지 않고 크기만 조회)

사용법: uv run python appendix/a7_drivaerml/get_data.py [저장 폴더]
    기본값 ~/cfd-runs/a7
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/datasets/neashton/drivaerml"
FILES = ["geo_parameters_all.csv", "force_mom_all.csv",
         "splits/manifest.json", "LICENSE.txt"]


def get(url, path):
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=60) as r:
        path.write_bytes(r.read())
    return path.stat().st_size, time.perf_counter() - start


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1
               else "~/cfd-runs/a7").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in FILES:
        size, sec = get(f"{REPO}/resolve/main/{f}", out / Path(f).name)
        total += size
        print(f"{f:<26} {size / 1e3:8.1f} KB  {sec:.1f}초")
    print(f"합계 {total / 1e3:.1f} KB -> {out}")

    url = REPO.replace("/datasets/", "/api/datasets/") + "/tree/main/run_1"
    with urllib.request.urlopen(url, timeout=60) as r:
        items = json.load(r)
    print("\nrun_1 폴더 (형상 하나)의 파일 크기")
    for it in items:
        if it["type"] == "file":
            size = it["size"]
            text = (f"{size / 1e6:10.1f} MB" if size >= 1e6
                    else f"{size:10d} B")
            print(f"  {it['path'].split('/')[-1]:<28} {text}")
