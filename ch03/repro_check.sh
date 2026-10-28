#!/bin/bash
# 책 03-5 재현성 검사
# 기준 형상(H 2 mm, e 0.6 mm, 리브 8개, U 0.1 m/s)을
#  1) 코어 하나로 세 번 해석해 결과가 비트 단위까지 같은지 보고,
#  2) 코어 1, 2, 4, 6개로 병렬 해석해 결과가 얼마나 달라지는지 보고,
#     코어 4개로 한 번 더 해석해 같은 코어 수면 결과가 같은지 본다.
#
# 사용법: bash ch03/repro_check.sh [작업 폴더]   (기본 ~/cfd-runs/repro)

. /opt/openfoam14/etc/bashrc

CH03=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${1:-$HOME/cfd-runs/repro}

if [ -e "$RUN_DIR" ]; then
    echo "이미 있습니다: $RUN_DIR (지우고 다시 실행하세요)"
    exit 1
fi
mkdir -p "$RUN_DIR"

make_case() {   # 템플릿 복사와 기준 형상 격자
    cp -r "$CH03/template" "$1"
    python3 "$CH03/channel_mesh.py" 2 0.6 8 \
        "$1/system/blockMeshDict" > /dev/null
    (cd "$1" && blockMesh > log.blockMesh 2>&1)
}

summary() {     # 이름, 케이스 폴더, 걸린 시간
    python3 - "$CH03" "$@" <<'EOF'
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from extract import extract
name, case, sec = sys.argv[2], Path(sys.argv[3]), float(sys.argv[4])
r = extract(case)
# 마지막 반복의 온도장 파일 지문. 병렬 해석은 조각 파일을 이어 붙여 계산한다
it = str(r["iterations"])
if (case / it).exists():
    files = [case / it / "T"]
else:
    files = sorted(case.glob(f"processor*/{it}/T"))
md5 = hashlib.md5(b"".join(f.read_bytes() for f in files)).hexdigest()[:8]
pad = " " * (11 - sum(2 if ord(c) > 0x1100 else 1 for c in name))  # 한글은 2칸
print(f"{name}{pad}반복 {it:>3}  압력손실 {r['dp']:.6e}"
      f"  최고 {r['dT_max']:.4f}  {sec:4.1f}초  {md5}")
EOF
}

echo "== 같은 조건 세 번 (코어 1개)"
for i in 1 2 3; do
    CASE=$RUN_DIR/serial$i
    make_case "$CASE"
    cd "$CASE" || exit 1
    s=$(date +%s.%N)
    foamRun > log.foamRun 2>&1
    summary "반복$i" "$CASE" "$(echo "$(date +%s.%N) - $s" | bc)"
done

echo "== 코어 수 바꾸기 (scotch 분할)"
for np in 1 2 4 6 4; do
    CASE=$RUN_DIR/np$np
    label="코어$np"
    if [ -e "$CASE" ]; then     # 같은 코어 수로 두 번째 해석
        CASE=${CASE}_again
        label="코어$np 다시"
    fi
    make_case "$CASE"
    cd "$CASE" || exit 1
    cat > system/decomposeParDict <<EOF
FoamFile
{
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
numberOfSubdomains  $np;
method              scotch;
EOF
    s=$(date +%s.%N)
    if [ "$np" = 1 ]; then
        foamRun > log.foamRun 2>&1
    else
        decomposePar > log.decomposePar 2>&1
        mpirun -np "$np" foamRun -parallel > log.foamRun 2>&1
    fi
    summary "$label" "$CASE""$(echo "$(date +%s.%N) - $s" | bc)"
done
