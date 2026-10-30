"""책 부록 A-4: 리브 채널 아래에 냉각판(고체)을 붙여 복합 열전달(CHT)을 푼다.

3장의 기준 형상(H 2, e 0.6, N 8, U 0.1) 아래에 두께 TS의 판을 채널 길이 전체에
붙인다. 판 아랫면의 가열 구간(x 10~70 mm)에 3장과 같은 열유속 10 kW/m²를
주고, 나머지 바깥면은 단열이다. 판 윗면은 물과 맞닿아 온도와 열유속을 주고받는다.

같은 솔버(fluid 모듈, 물 밀도 일정)로 판이 없는 경우도 풀어 판의 효과만 본다.
- fluid: 판 없음. 3장처럼 물이 닿는 아랫벽에 열유속을 바로 준다
- cht:   판 있음. foamMultiRun으로 물(fluid)과 판(solid)을 함께 푼다

사용법: uv run python appendix/a4_3d_cht/cht2d.py [작업 폴더] [재질]
    기본값 ~/cfd-runs/a4/cht  al      (재질: al 알루미늄, steel 스테인리스)
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ch03"))
from channel_mesh import (ASPECT, CELLS_H, DEPTH, L_HEAT, L_IN,  # noqa: E402
                          L_OUT, rib_spans)

H, E, N, U = 2.0, 0.6, 8, 0.1     # 03-2의 기준 형상
TS, NS = 2.0, 15                   # 판 두께 [mm], 두께 방향 셀 수
Q = 1e4                            # 열유속 [W/m²]
T_IN = 293.15
RHO, CP, MU = 998.0, 4182.0, 9.98e-4
PR = MU / RHO / 1.4376e-7          # 3장의 열확산계수와 같게: Pr = nu / alpha
SOLIDS = {"al": (2700, 896, 167.0), "steel": (8000, 500, 16.0)}  # rho, Cv, k
ITER = 4000


def header(cls, obj):
    return (f"FoamFile\n{{\n    format      ascii;\n    class       {cls};\n"
            f"    object      {obj};\n}}\n\n")


def mesh(solid):
    """blockMeshDict. 판이 있으면 맨 아래 줄 블록을 solid 영역으로 둔다."""
    ribs = rib_spans(N)
    L = L_IN + L_HEAT + L_OUT
    xs = sorted({0.0, L_IN, L_IN + L_HEAT, L} | {x for r in ribs for x in r})
    ys = ([-TS] if solid else []) + [0.0, H - E, H]
    nb, nr = len(xs) - 1, len(ys)
    rib = [any(abs(xs[i] - a) < 1e-9 for a, _ in ribs) for i in range(nb)]
    dy = H / CELLS_H
    ncx = [max(1, round((xs[i + 1] - xs[i]) / (ASPECT * dy)))
           for i in range(nb)]
    f0 = 1 if solid else 0                       # 유체 첫 줄의 번호
    ncy = [NS] * f0 + [max(2, round((ys[j + 1] - ys[j]) / dy))
                       for j in range(f0, f0 + 2)]

    def v(i, j, k):
        return k * nr * (nb + 1) + j * (nb + 1) + i

    verts = [(x, y, z) for z in (0, DEPTH) for y in ys for x in xs]
    blocks, P = [], {n: [] for n in ("inlet", "outlet", "lowerWall",
                                     "heatedWall", "upperWall", "ribWall",
                                     "heater", "solidWall")}
    for i in range(nb):
        heated = L_IN <= xs[i] < L_IN + L_HEAT
        for j in range(nr - 1):
            if j == f0 + 1 and rib[i]:
                continue
            c = [v(i, j, 0), v(i + 1, j, 0), v(i + 1, j + 1, 0),
                 v(i, j + 1, 0), v(i, j, 1), v(i + 1, j, 1),
                 v(i + 1, j + 1, 1), v(i, j + 1, 1)]
            zone = ("solid" if j < f0 else "fluid") if solid else ""
            blocks.append((c, zone, ncx[i], ncy[j]))
            bottom, top = (c[0], c[1], c[5], c[4]), (c[3], c[7], c[6], c[2])
            left, right = (c[0], c[4], c[7], c[3]), (c[1], c[2], c[6], c[5])
            if j < f0:                                    # 판
                P["heater" if heated else "solidWall"].append(bottom)
                if i == 0:
                    P["solidWall"].append(left)
                if i == nb - 1:
                    P["solidWall"].append(right)
                continue
            if j == f0:
                if not solid:
                    P["heatedWall" if heated else "lowerWall"].append(bottom)
                if rib[i]:
                    P["ribWall"].append(top)
            else:
                P["upperWall"].append(top)
                if i > 0 and rib[i - 1]:
                    P["ribWall"].append(left)
                if i < nb - 1 and rib[i + 1]:
                    P["ribWall"].append(right)
            if i == 0:
                P["inlet"].append(left)
            if i == nb - 1:
                P["outlet"].append(right)

    out = [header("dictionary", "blockMeshDict"), "units [mm];\n",
           "vertices\n("]
    out += [f"    ({x:g} {y:g} {z:g})" for x, y, z in verts]
    out += [");\n", "blocks\n("]
    out += [f"    hex ({' '.join(map(str, c))}) {z} ({nx} {ny} 1)"
            " simpleGrading (1 1 1)" for c, z, nx, ny in blocks]
    out += [");\n", "defaultPatch\n{\n    name frontAndBack;\n"
            "    type empty;\n}\n", "boundary\n("]
    for name, faces in P.items():
        if not faces:
            continue
        kind = "patch" if name in ("inlet", "outlet") else "wall"
        out += [f"    {name}\n    {{\n        type {kind};\n        faces\n"
                "        ("]
        out += [f"            ({' '.join(map(str, f))})" for f in faces]
        out += ["        );\n    }"]
    out += [");"]
    return "\n".join(out) + "\n"


def field(cls, obj, dim, internal, bc):
    body = "\n".join(f"    {k}\n    {{\n" + "".join(
        f"        {a:<16}{b};\n" for a, b in v.items()) + "    }"
        for k, v in bc.items())
    return (header(cls, obj) + f"dimensions      [{dim}];\n\n"
            f"internalField   uniform {internal};\n\nboundaryField\n{{\n"
            f"{body}\n    #includeEtc \"caseDicts/setConstraintTypes\"\n}}\n")


def fluid_fields(solid):
    wall = '"(.*Wall|fluid_to_solid)"'
    t_bc = {"inlet": {"type": "fixedValue", "value": f"uniform {T_IN}"},
            "outlet": {"type": "inletOutlet", "inletValue": f"uniform {T_IN}",
                       "value": f"uniform {T_IN}"},
            '".*Wall"': {"type": "zeroGradient"}}
    if solid:
        t_bc["fluid_to_solid"] = {"type": "coupledTemperature",
                                  "value": f"uniform {T_IN}"}
    else:
        t_bc["heatedWall"] = {"type": "externalTemperature",
                              "q": f"constant {Q:g}",
                              "value": f"uniform {T_IN}"}
    return {
        "U": field("volVectorField", "U", "velocity", "(0 0 0)", {
            "inlet": {"type": "fixedValue", "value": f"uniform ({U} 0 0)"},
            "outlet": {"type": "zeroGradient"},
            wall: {"type": "noSlip"}}),
        "p_rgh": field("volScalarField", "p_rgh", "pressure", 0, {
            "inlet": {"type": "fixedFluxPressure", "value": "uniform 0"},
            "outlet": {"type": "fixedValue", "value": "uniform 0"},
            wall: {"type": "fixedFluxPressure", "value": "uniform 0"}}),
        "p": field("volScalarField", "p", "pressure", "1e5", {
            '".*"': {"type": "calculated", "value": "uniform 1e5"}}),
        "T": field("volScalarField", "T", "temperature", T_IN, t_bc),
    }


FLUID_PROPS = f"""thermoType
{{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          eConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleInternalEnergy;
}}

mixture
{{
    specie
    {{
        molWeight   18;
    }}
    equationOfState
    {{
        rho         {RHO:g};
    }}
    thermodynamics
    {{
        Cv          {CP:g};
        hf          0;
    }}
    transport
    {{
        mu          {MU:g};
        Pr          {PR:.4f};
    }}
}}
"""

FLUID_SCHEMES = """ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,e)      bounded Gauss linearUpwind grad(e);
    div(phi,K)      bounded Gauss linear;
    div(phi,(p|rho)) bounded Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""

FLUID_SOLUTION = """solvers
{
    rho
    {
        solver          diagonal;
    }
    p_rgh
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-9;
        relTol          0.01;
    }
    "(U|e)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-10;
        relTol          0.1;
    }
}

PIMPLE
{
    momentumPredictor yes;
    nCorrectors     1;
    nNonOrthogonalCorrectors 0;
}

relaxationFactors
{
    fields
    {
        p_rgh           0.7;
    }
    equations
    {
        U               0.7;
        e               0.9;
    }
}
"""

SOLID_SCHEMES = """ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""

SOLID_SOLUTION = """solvers
{
    e
    {
        solver          GAMG;
        smoother        symGaussSeidel;
        tolerance       1e-10;
        relTol          0.1;
    }
}

PIMPLE
{
    nNonOrthogonalCorrectors 0;
}

relaxationFactors
{
    equations
    {
        ".*"            1;
    }
}
"""


def functions(solid):
    reg = " region fluid;" if solid else ""
    sreg = "region=solid, " if solid else ""
    wall = "fluid_to_solid" if solid else "heatedWall"
    fr = f"region=fluid, " if solid else ""
    out = f"""functions
{{
    pSection
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");{reg}
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        sampledSurface
        {{
            type        cutPlane;
            point       (0.005 0 0);
            normal      (1 0 0);
            interpolate false;
        }}
        operation       areaAverage;
        fields          (p);
    }}
    #includeFunc patchAverage({fr}funcName=wallTmax, patch={wall}, fields=(T), operation=max)
    #includeFunc patchAverage({fr}funcName=outletT, patch=outlet, fields=(T), operation=average, weightField=phi)
"""
    if solid:
        out += (f"    #includeFunc patchAverage({sreg}funcName=heaterTmax, "
                "patch=heater, fields=(T), operation=max)\n"
                f"    #includeFunc patchAverage({sreg}funcName=heaterTavg, "
                "patch=heater, fields=(T))\n")
    else:
        out += ("    #includeFunc patchAverage(funcName=wallTavg, "
                "patch=heatedWall, fields=(T))\n")
    return out + "}\n"


def control(solid):
    head = ("regionSolvers\n{\n    fluid           fluid;\n"
            "    solid           solid;\n}\n" if solid else
            "solver          fluid;\n")
    return (header("dictionary", "controlDict") + head +
            f"\nstartFrom       startTime;\nstartTime       0;\n"
            f"stopAt          endTime;\nendTime         {ITER};\n"
            f"deltaT          1;\nwriteControl    timeStep;\n"
            f"writeInterval   {ITER // 2};\npurgeWrite      0;\n"
            f"writeFormat     ascii;\nwritePrecision  8;\n"
            f"timeFormat      general;\nrunTimeModifiable false;\n\n"
            + functions(solid))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_case(case, solid, material="al"):
    subprocess.run(["rm", "-rf", str(case)], check=True)
    fdir = "fluid/" if solid else ""
    write(case / "system/blockMeshDict", mesh(solid))
    write(case / "system/controlDict", control(solid))
    for name, text in fluid_fields(solid).items():
        write(case / f"0/{fdir}{name}", text)
    write(case / f"constant/{fdir}physicalProperties",
          header("dictionary", "physicalProperties") + FLUID_PROPS)
    write(case / f"constant/{fdir}momentumTransport",
          header("dictionary", "momentumTransport") +
          "simulationType  laminar;\n")
    write(case / f"constant/{fdir}g", header("uniformDimensionedVectorField",
                                            "g") +
          "dimensions      [acceleration];\nvalue           (0 0 0);\n")
    write(case / f"constant/{fdir}pRef",
          header("uniformDimensionedScalarField", "pRef") +
          "dimensions      [pressure];\nvalue           1e5;\n")
    write(case / f"system/{fdir}fvSchemes",
          header("dictionary", "fvSchemes") + FLUID_SCHEMES)
    write(case / f"system/{fdir}fvSolution",
          header("dictionary", "fvSolution") + FLUID_SOLUTION)
    if not solid:
        return
    rho, cv, k = SOLIDS[material]
    write(case / "system/fvSolution", header("dictionary", "fvSolution") +
          "PIMPLE\n{\n    nOuterCorrectors 1;\n}\n")
    write(case / "system/solid/fvSchemes",
          header("dictionary", "fvSchemes") + SOLID_SCHEMES)
    write(case / "system/solid/fvSolution",
          header("dictionary", "fvSolution") + SOLID_SOLUTION)
    write(case / "constant/solid/physicalProperties",
          header("dictionary", "physicalProperties") +
          f"thermoType      constSolidThermo;\n\nrho\n{{\n    type uniform;\n"
          f"    value {rho};\n}}\n\nCv\n{{\n    type uniform;\n"
          f"    value {cv};\n}}\n\nkappa\n{{\n    type uniform;\n"
          f"    value {k};\n}}\n")
    write(case / "0/solid/T", field("volScalarField", "T", "temperature",
                                    T_IN, {
        "heater": {"type": "externalTemperature", "q": f"constant {Q:g}",
                   "value": f"uniform {T_IN}"},
        "solidWall": {"type": "zeroGradient"},
        "solid_to_fluid": {"type": "coupledTemperature",
                           "value": f"uniform {T_IN}"}}))


def sh(case, *cmd):
    with open(case / f"log.{cmd[0]}", "w") as f:
        subprocess.run(list(cmd), cwd=case, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


def last(case, name):
    """함수 객체 기록의 마지막 값과 절반 시점 값."""
    dat = next((case / "postProcessing").rglob(f"{name}/0/*.dat"))
    rows = [r.split() for r in dat.read_text().splitlines()
            if r.strip() and not r.startswith("#")]
    half = next(r for r in rows if int(float(r[0])) == ITER // 2)
    return float(rows[-1][1]), float(half[1])


def run(case, solid, material="al"):
    make_case(case, solid, material)
    start = time.perf_counter()
    sh(case, "blockMesh")
    if solid:
        sh(case, "splitMeshRegions", "-cellZones", "all")
        sh(case, "foamMultiRun")
    else:
        sh(case, "foamRun")
    sec = time.perf_counter() - start
    res = {"seconds": sec}
    for name in ["pSection", "wallTmax", "outletT"] + (
            ["heaterTmax", "heaterTavg"] if solid else ["wallTavg"]):
        v, h = last(case, name)
        off = 1e5 if name == "pSection" else T_IN
        res[name] = v - off
        res[name + "_half"] = h - off
    return res


if __name__ == "__main__":
    work = Path(sys.argv[1] if len(sys.argv) > 1
                else "~/cfd-runs/a4/cht").expanduser()
    material = sys.argv[2] if len(sys.argv) > 2 else "al"
    work.mkdir(parents=True, exist_ok=True)
    dt_energy = Q * L_HEAT * 1e-3 / (RHO * CP * U * H * 1e-3)
    print(f"에너지 수지로 정해지는 출구 온도 상승 {dt_energy:.4f} K")
    for label, solid in [("판 없음", False), ("판 있음", True)]:
        r = run(work / ("cht_" + material if solid else "fluid"), solid,
                material)
        print(f"\n[{label}] {r['seconds']:.0f}초, 반복 {ITER}")
        print(f"  압력손실 {r['pSection'] / RHO:.5f} m2/s2, 출구 온도 상승 "
              f"{r['outletT']:.4f} K")
        print(f"  물이 닿는 면 최고 온도 상승 {r['wallTmax']:.3f} K "
              f"(절반 시점 {r['wallTmax_half']:.3f})")
        if solid:
            print(f"  판 아랫면(가열면) 최고 {r['heaterTmax']:.3f} K, "
                  f"평균 {r['heaterTavg']:.3f} K "
                  f"(절반 시점 {r['heaterTmax_half']:.3f})")
        else:
            print(f"  가열면 평균 {r['wallTavg']:.3f} K")
