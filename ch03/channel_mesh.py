"""책 03-1: 리브 냉각 채널의 형상 변수로 blockMeshDict를 만든다.

아래 벽(가열면)은 평평하고, 위 벽(덮개)에 사각 리브 N개가 매달려 있다.
리브가 흐름을 가열면 쪽으로 밀어 냉각을 돕는 대신 압력손실이 커진다.

사용법: uv run python ch03/channel_mesh.py H e N 출력파일
    H  채널 높이 [mm]   e  리브 높이 [mm]   N  리브 개수
예:     uv run python ch03/channel_mesh.py 2 1 8 system/blockMeshDict
"""
import sys
from pathlib import Path

L_IN, L_HEAT, L_OUT = 10.0, 60.0, 10.0   # 입구·가열·출구 구간 길이 [mm]
W_RIB = 0.5         # 리브 폭 [mm]
DEPTH = 0.1         # z 두께 [mm] (2D라 결과에 영향 없음)
CELLS_H = 30        # 채널 높이 방향 셀 수
ASPECT = 2          # 셀 가로세로비 (셀 길이 / 셀 높이)

HEADER = """\
FoamFile
{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

units [mm];
"""


def rib_spans(N):
    """가열 구간을 N칸으로 나누고 칸마다 가운데에 리브를 하나 둔다."""
    pitch = L_HEAT / N
    centers = [L_IN + pitch * (i + 0.5) for i in range(N)]
    return [(c - W_RIB / 2, c + W_RIB / 2) for c in centers]


def block_mesh_dict(H, e, N):
    ribs = rib_spans(N)
    L = L_IN + L_HEAT + L_OUT
    xs = sorted({0.0, L_IN, L_IN + L_HEAT, L} | {x for r in ribs for x in r})
    ys = [0.0, H - e, H]          # 아래 층: 주 유로, 위 층: 리브 사이 공간
    nx_b = len(xs) - 1
    rib = [any(abs(xs[i] - a) < 1e-9 for a, _ in ribs) for i in range(nx_b)]

    dy = H / CELLS_H              # 셀 높이를 기준으로 칸 수를 정한다
    ncx = [max(1, round((xs[i + 1] - xs[i]) / (ASPECT * dy)))
           for i in range(nx_b)]
    ncy = [max(2, round((ys[j + 1] - ys[j]) / dy)) for j in range(2)]

    def v(i, j, k):               # 꼭짓점 번호: x 위치 i, 높이 j, 앞뒤 k
        return k * 3 * (nx_b + 1) + j * (nx_b + 1) + i

    vertices = [(x, y, z) for z in (0, DEPTH) for y in ys for x in xs]
    blocks = []
    patches = {name: [] for name in
               ("inlet", "outlet", "lowerWall", "heatedWall",
                "upperWall", "ribWall")}
    for i in range(nx_b):
        heated = L_IN <= xs[i] < L_IN + L_HEAT
        for j in range(2):
            if j == 1 and rib[i]:
                continue          # 리브가 차지한 자리는 블록을 만들지 않는다
            c = [v(i, j, 0), v(i + 1, j, 0), v(i + 1, j + 1, 0),
                 v(i, j + 1, 0), v(i, j, 1), v(i + 1, j, 1),
                 v(i + 1, j + 1, 1), v(i, j + 1, 1)]
            blocks.append((c, ncx[i], ncy[j]))
            bottom, top = (c[0], c[1], c[5], c[4]), (c[3], c[7], c[6], c[2])
            left, right = (c[0], c[4], c[7], c[3]), (c[1], c[2], c[6], c[5])
            if j == 0:
                patches["heatedWall" if heated else "lowerWall"].append(bottom)
                if rib[i]:
                    patches["ribWall"].append(top)      # 리브 아랫면
            else:
                patches["upperWall"].append(top)
                if i > 0 and rib[i - 1]:
                    patches["ribWall"].append(left)     # 리브 옆면
                if i < nx_b - 1 and rib[i + 1]:
                    patches["ribWall"].append(right)
            if i == 0:
                patches["inlet"].append(left)
            if i == nx_b - 1:
                patches["outlet"].append(right)

    out = [HEADER, "vertices\n("]
    out += [f"    ({x:g} {y:g} {z:g})" for x, y, z in vertices]
    out += [");\n", "blocks\n("]
    out += [f"    hex ({' '.join(map(str, c))}) ({nx} {ny} 1)"
            " simpleGrading (1 1 1)" for c, nx, ny in blocks]
    out += [");\n", "defaultPatch\n{\n    name frontAndBack;\n"
            "    type empty;\n}\n", "boundary\n("]
    for name, faces in patches.items():
        kind = "patch" if name in ("inlet", "outlet") else "wall"
        out += [f"    {name}\n    {{\n        type {kind};\n        faces\n"
                "        ("]
        out += [f"            ({' '.join(map(str, f))})" for f in faces]
        out += ["        );\n    }"]
    out += [");"]
    cells = sum(nx * ny for _, nx, ny in blocks)
    return "\n".join(out) + "\n", cells


if __name__ == "__main__":
    H, e, N = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    text, cells = block_mesh_dict(H, e, N)
    Path(sys.argv[4]).write_text(text)
    print(f"H {H} mm, e {e} mm, 리브 {N}개 (간격 {L_HEAT / N:g} mm)"
          f" -> 셀 {cells}개")
