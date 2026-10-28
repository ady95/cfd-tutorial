#!/bin/bash
# 책 03-1 형상 범위 끝의 격자 검사
# 설계 변수 범위의 양 끝 형상 네 개와 기준 형상으로 격자를 만들고
# checkMesh 결과를 한 줄씩 요약한다. 해석(foamRun)은 하지 않는다.
#
# 사용법: bash ch03/mesh_check.sh [작업 폴더]   (기본 ~/cfd-runs/rib-check)

. /opt/openfoam14/etc/bashrc

CH03=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${1:-$HOME/cfd-runs/rib-check}
mkdir -p "$RUN_DIR"

#        H    e    N     (채널 높이 mm, 리브 높이 mm, 리브 개수)
for g in "1.5 0.3 4" "1.5 0.6 15" "2.5 0.5 15" "2.5 1.0 4" "2 0.6 8"; do
    set -- $g
    CASE=$RUN_DIR/H$1_e$2_N$3
    rm -rf "$CASE"
    cp -r "$CH03/template" "$CASE"
    python3 "$CH03/channel_mesh.py" "$1" "$2" "$3" \
        "$CASE/system/blockMeshDict" > /dev/null
    cd "$CASE" || exit 1
    blockMesh > log.blockMesh 2>&1
    checkMesh > log.checkMesh 2>&1

    cells=$(grep -E "^ +cells:" log.checkMesh | awk '{print $2}')
    aspect=$(grep "Max aspect ratio" log.checkMesh | awk '{print $5}')
    ok=$(grep -c "^Mesh OK" log.checkMesh)
    echo "H $1  e $2  N $3   셀 $cells  최대 가로세로비 $aspect" \
         "  $([ "$ok" = 1 ] && echo 'Mesh OK' || echo '검사 실패')"
done
