#!/bin/bash
# 책 02-5 격자 독립성 검사: channel2d를 격자 네 개로 해석한다.
# 높이 방향 칸 수 ny를 10, 20, 40, 80으로 두 배씩 늘리고,
# 길이 방향 nx = 10 ny로 함께 늘려 셀 모양(가로세로비 5)을 유지한다.
# 촘촘한 격자에서도 반복 오차가 섞이지 않게 수렴 기준을 1e-7로 낮춘다.
#
# 사용법: bash ch02/grid_run.sh [작업 폴더]   (기본 ~/cfd-runs/grid)

. /opt/openfoam14/etc/bashrc

SRC=$(cd "$(dirname "$0")" && pwd)/channel2d
RUN_DIR=${1:-$HOME/cfd-runs/grid}

if [ -e "$RUN_DIR" ]; then
    echo "이미 있습니다: $RUN_DIR (지우고 다시 실행하세요)"
    exit 1
fi
mkdir -p "$RUN_DIR"

for ny in 10 20 40 80; do
    nx=$((ny * 10))
    CASE=$RUN_DIR/ny$ny
    cp -r "$SRC" "$CASE"
    cd "$CASE" || exit 1

    sed -i "s/^nx .*/nx  $nx;/; s/^ny .*/ny  $ny;/" system/blockMeshDict
    for f in p U; do
        foamDictionary system/fvSolution \
            -entry SIMPLE/residualControl/$f -set 1e-7 > /dev/null
    done
    foamDictionary system/controlDict -entry endTime -set 20000 > /dev/null
    ./Allrun > /dev/null

    it=$(grep -o "converged in [0-9]*" log.foamRun | grep -o "[0-9]*$")
    sec=$(grep ExecutionTime log.foamRun | tail -1 | awk '{print $3}')
    echo "ny $ny  nx $nx  반복 $it  시간 $sec s"
done
