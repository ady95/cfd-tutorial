#!/bin/bash
# 책 부록 A-1: 기준점 재현 검사
#   1) 환경 점검 (check_env.py)
#   2) 기준 형상 CFD 한 건 (03-5의 반복1과 같은 조건, 코어 하나)
#      -> 반복 수, 압력손실, 최고 온도, 온도장 파일 지문을 책 값과 비교
#   3) 최종 대리모델(models/surrogate_v1)의 기준 설계 예측을 책 값과 비교
#
# 사용법 (저장소 루트에서):
#   bash appendix/a1_env/ref_check.sh [작업 폴더]   (기본 /tmp/ref_check)
#   호스트에서는 uv run이, 컨테이너에서는 /opt/venv의 python이 쓰인다

. /opt/openfoam14/etc/bashrc
cd "$(dirname "$0")/../.." || exit 1
RUN=${1:-/tmp/ref_check}
PY="python"
command -v uv > /dev/null && [ -z "$UV_PROJECT_ENVIRONMENT" ] && PY="uv run python"

echo "== 1. 환경"
$PY appendix/a1_env/check_env.py | tail -4

echo "== 2. 기준 형상 CFD (책 03-5 반복1)"
rm -rf "$RUN" && mkdir -p "$RUN"
cp -r ch03/template "$RUN/base"
python3 ch03/channel_mesh.py 2 0.6 8 "$RUN/base/system/blockMeshDict" > /dev/null
(cd "$RUN/base" && blockMesh > log.blockMesh 2>&1 && foamRun > log.foamRun 2>&1)
$PY - "$RUN/base" <<'EOF'
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, "ch03")
from extract import extract
case = Path(sys.argv[1])
r = extract(case)
md5 = hashlib.md5((case / str(r["iterations"]) / "T").read_bytes()).hexdigest()
got = (r["iterations"], f"{r['dp']:.6e}", f"{r['dT_max']:.4f}", md5[:8])
book = (124, "4.227779e-02", "7.6666", "3b986601")
for name, g, b in zip(["반복 수", "압력손실", "최고 온도 상승", "온도장 지문"],
                      got, book):
    w = sum(2 if ord(c) > 0x1100 else 1 for c in name)
    print(f"{name}{' ' * (16 - w)}책 {str(b):<14} 지금 {str(g):<14} "
          f"{'OK' if g == b else '다름'}")
EOF

echo "== 3. 대리모델 예측 (models/surrogate_v1, 기준 설계)"
$PY - <<'EOF'
import sys
sys.path.insert(0, "ch06")
import numpy as np
from common import load_mlp
p = load_mlp()(np.array([[2.0, 0.3, 8, 0.1]]))[0]
got = [f"{v:.6g}" for v in p]
book = ["0.042283", "7.66745", "5.35671"]
for name, g, b in zip(["dp", "dT_max", "dT_avg"], got, book):
    print(f"{name:<16}책 {b:<14} 지금 {g:<14} {'OK' if g == b else '다름'}")
EOF
