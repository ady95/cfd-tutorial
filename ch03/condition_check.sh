#!/bin/bash
# 책 03-2 운전 조건 실험
# 기준 형상(H 2 mm, e 0.6 mm, 리브 8개)에서 열유속·입구 온도·유속을
# 하나씩 바꿔 해석하고, 압력과 온도 상승을 한 줄씩 요약한다.
#
# 사용법: bash ch03/condition_check.sh [작업 폴더]   (기본 ~/cfd-runs/cond)

. /opt/openfoam14/etc/bashrc

CH03=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${1:-$HOME/cfd-runs/cond}

if [ -e "$RUN_DIR" ]; then
    echo "이미 있습니다: $RUN_DIR (지우고 다시 실행하세요)"
    exit 1
fi
mkdir -p "$RUN_DIR"

#        이름     유속 U  열유속 q  입구 온도 Tin
for c in "base    0.1   1e4   293.15" \
         "q2x     0.1   2e4   293.15" \
         "Tin+5   0.1   1e4   298.15" \
         "U0.05   0.05  1e4   293.15" \
         "U0.15   0.15  1e4   293.15"; do
    set -- $c
    CASE=$RUN_DIR/$1
    cp -r "$CH03/template" "$CASE"
    python3 "$CH03/channel_mesh.py" 2 0.6 8 \
        "$CASE/system/blockMeshDict" > /dev/null
    cd "$CASE" || exit 1

    # 경계조건 값 바꾸기: 열유속은 기울기 q / k 로 넣는다 (k = 0.6 W/m/K)
    grad=$(python3 -c "print(f'{$3 / 0.6:.6g}')")
    foamDictionary 0/U -entry boundaryField/inlet/value \
        -set "uniform ($2 0 0)" > /dev/null
    foamDictionary 0/T -entry boundaryField/heatedWall/gradient \
        -set "uniform $grad" > /dev/null
    foamDictionary 0/T -entry boundaryField/inlet/value \
        -set "uniform $4" > /dev/null
    foamDictionary 0/T -entry internalField -set "uniform $4" > /dev/null
    ./Allrun > /dev/null

    it=$(grep -o "converged in [0-9]*" log.foamRun | grep -o "[0-9]*$")
    p_in=$(tail -1 postProcessing/patchAverage*/0/*.dat | awk '{print $2}')
    t_out=$(tail -1 postProcessing/outletT/0/*.dat | awk '{print $2}')
    t_max=$(tail -1 postProcessing/wallTmax/0/*.dat | awk '{print $2}')
    t_avg=$(tail -1 postProcessing/wallTavg/0/*.dat | awk '{print $2}')
    awk -v n="$1" -v it="$it" -v p="$p_in" -v tin="$4" -v to="$t_out" \
        -v tm="$t_max" -v ta="$t_avg" 'BEGIN {
        printf "%-5s  반복 %3s  입구 p %.5f 온도 상승[K] 출구 %.4f" \
               " 최고 %6.3f 평균 %6.3f\n", n, it, p, to - tin,
               tm - tin, ta - tin }'
done
