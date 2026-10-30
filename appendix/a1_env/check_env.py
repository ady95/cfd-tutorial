"""책 부록 A-1: 지금 환경이 책의 실측 환경과 같은지 점검한다.

1) OpenFOAM: 패키지 버전과 빌드 문자열
2) Python 패키지: 설치된 버전과 uv.lock의 버전을 하나씩 비교
3) GPU: PyTorch가 GPU를 쓰는지

사용법: uv run python appendix/a1_env/check_env.py
"""
import os
import platform
import subprocess
import tomllib
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BOOK = {"openfoam14": "20261008", "openfoam_build": "14-a4867fc914c1",
        "python": "3.12"}
KEY = ["torch", "numpy", "pandas", "scikit-learn", "xgboost", "optuna",
       "botorch", "neuraloperator", "nvidia-physicsnemo", "pyvista",
       "fastapi", "uvicorn"]


def dpkg_version(pkg):
    p = subprocess.run(["dpkg-query", "-W", "-f=${Version}", pkg],
                       capture_output=True, text=True)
    return p.stdout.strip() or None


def mark(ok):
    return "OK" if ok else "다름"


def pad(text, width):
    """한글은 2칸으로 세어 오른쪽을 공백으로 채운다."""
    text = str(text)
    return text + " " * (width - sum(2 if ord(c) > 0x1100 else 1
                                     for c in text))


build = Path(os.environ.get("WM_PROJECT_DIR", "/opt/openfoam14")) / ".build"
rows = [
    ("openfoam14 패키지", BOOK["openfoam14"], dpkg_version("openfoam14")),
    ("OpenFOAM 빌드", BOOK["openfoam_build"],
     build.read_text().strip() if build.exists() else None),
    ("Python", BOOK["python"], platform.python_version()),
]
lock = tomllib.loads((REPO / "uv.lock").read_text())
locked = {p["name"]: p["version"] for p in lock["package"]}
for name in KEY:
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        have = None
    rows.append((name, locked.get(name), have))

print(f"{pad('항목', 20)}{pad('책 기준', 18)}{pad('지금', 18)}판정")
bad = 0
for name, want, have in rows:
    ok = have is not None and want is not None and (
        have.startswith(want) if name == "Python" else have == want)
    bad += not ok
    print(f"{pad(name, 20)}{pad(want, 18)}{pad(have, 18)}{mark(ok)}")

# 잠금 파일의 나머지 패키지까지 모두 비교
diff = []
for name, want in locked.items():
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        continue                 # 다른 플랫폼 전용 패키지는 설치되지 않는다
    if have != want and name != "cfd-tutorial":
        diff.append(f"{name} {want} -> {have}")
print(f"\nuv.lock 패키지 {len(locked)}개 중 버전이 다른 것: {len(diff)}개")
for d in diff[:10]:
    print("  " + d)

import torch  # noqa: E402

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음"
print(f"\nPyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {gpu}")
print("판정:", "책과 같은 환경" if bad == 0 and not diff
      else f"다른 항목 {bad + len(diff)}개")
