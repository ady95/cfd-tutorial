#!/bin/bash
# 책 03-3 결과 추출과 에너지 수지 검증
# 설계 변수 범위 끝의 형상·유속 다섯 개를 해석하고 extract.py 로 값을 뽑아
# 출구 온도 상승이 에너지 수지와 맞는지 한 줄씩 보여 준다.
#
# 사용법: bash ch03/extract_check.sh [작업 폴더]   (기본 ~/cfd-runs/extract)

. /opt/openfoam14/etc/bashrc

CH03=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${1:-$HOME/cfd-runs/extract}

if [ -e "$RUN_DIR" ]; then
    echo "이미 있습니다: $RUN_DIR (지우고 다시 실행하세요)"
    exit 1
fi
mkdir -p "$RUN_DIR"

# 온도 상승[K]: 출구 = 해석 결과, 수지 = 에너지 수지로 계산한 값
echo "H    e    N  U     반복  압력손실    출구     수지     최고"
#        H    e    N   U   (mm, mm, 개, m/s)
for c in "1.5  0.3  4  0.05" \
         "1.5  0.6  15 0.1" \
         "2    0.6  8  0.1" \
         "2.5  0.5  15 0.15" \
         "2.5  1.0  4  0.05"; do
    set -- $c
    CASE=$RUN_DIR/H$1_e$2_N$3_U$4
    cp -r "$CH03/template" "$CASE"
    python3 "$CH03/channel_mesh.py" "$1" "$2" "$3" \
        "$CASE/system/blockMeshDict" > /dev/null
    cd "$CASE" || exit 1
    foamDictionary 0/U -entry boundaryField/inlet/value \
        -set "uniform ($4 0 0)" > /dev/null
    ./Allrun > /dev/null

    python3 - "$CH03" "$CASE" "$c" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from extract import extract
r = extract(sys.argv[2])
it = r["iterations"] if r["converged"] else "실패"
print(f"{sys.argv[3]:<17}  {it:>4}  {r['dp']:.5f}  {r['dT_out']:7.4f}"
      f"  {r['dT_out_energy']:7.4f}  {r['dT_max']:7.3f}")
EOF
done
