#!/bin/bash
# 책 02-2 수렴 판정 실험
# pitzDailySteady 튜토리얼에서 residualControl(수렴 기준)을 지우고
# endTime(2000회)까지 실행한 뒤, 100회마다 저장된 결과에서
# 입구 평균 압력을 뽑는다.
#
# 사용법: bash ch02/convergence_check.sh [작업 폴더]   (기본 ~/cfd-runs)

. /opt/openfoam14/etc/bashrc

RUN_DIR=${1:-$HOME/cfd-runs}
CASE=$RUN_DIR/pitzDailySteady-full

if [ -e "$CASE" ]; then
    echo "이미 있습니다: $CASE (지우고 다시 실행하세요)"
    exit 1
fi

mkdir -p "$RUN_DIR"
cp -r "$FOAM_TUTORIALS/incompressibleFluid/pitzDailySteady" "$CASE"
cd "$CASE" || exit 1

foamDictionary system/fvSolution -entry SIMPLE/residualControl -remove
./Allrun

grep ExecutionTime log.foamRun | tail -1
foamPostProcess -func "patchAverage(patch=inlet, fields=(p))" \
    | grep -E "^Time =|areaAverage"
