"""책 부록 A-4 그림: 3D 채널의 폭 방향 온도와 냉각판 유무에 따른 가열면 온도.

왼쪽: 3D 채널 가열면 최고 온도 상승의 폭 방향 분포 (z 셀 20개, 40개)와 2D 값
오른쪽: 가열면을 따라간 온도 상승 (판 없음 / 알루미늄 판 / 스테인리스 판)

사용법: uv run python appendix/a4_3d_cht/plot_a4.py [작업 폴더]
    기본값 ~/cfd-runs/a4   (channel3d.py, cht2d.py를 먼저 실행)
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T_IN = 293.15
DT_2D = 7.6666                     # 03-2 기준 형상의 가열면 최고 온도 상승


def patch_profile(case, region, patch):
    """경계면 온도 상승을 x 위치마다 최댓값으로 (2D라 x마다 면 하나)."""
    foam = case / "case.foam"
    foam.touch()
    r = pv.POpenFOAMReader(str(foam))
    r.set_active_time_value(r.time_values[-1])
    r.enable_all_patch_arrays()
    d = r.read()
    block = d[region]["boundary"][patch] if region else d["boundary"][patch]
    x = block.cell_centers().points[:, 0] * 1e3
    t = block.cell_data["T"] - T_IN
    k = np.argsort(x)
    return x[k], t[k]


if __name__ == "__main__":
    work = Path(sys.argv[1] if len(sys.argv) > 1
                else "~/cfd-runs/a4").expanduser()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4))

    for name, label, style in [("3d", "3D, 20 cells across half width", "-"),
                               ("3d_nz40", "3D, 40 cells", "--")]:
        z, t = np.loadtxt(work / name / "wall_profile.csv", delimiter=",",
                          unpack=True)
        a.plot(z, t, style, color="#2a6f97", lw=2, label=label)
    a.axhline(DT_2D, color="#888888", lw=1.5, ls=":", label="2D (03-2)")
    a.set_xlabel("z (mm): 0 = channel centre, 5 = side wall")
    a.set_ylabel("max heated-wall temperature rise (K)")
    a.set_title("3D channel, 10 mm wide")
    a.grid(alpha=0.3)
    a.legend(frameon=False, loc="lower left")

    cases = [("fluid", None, "heatedWall", "no plate (heat flux on wall)",
              "#888888"),
             ("cht_al", "solid", "heater", "aluminium plate 2 mm, bottom",
              "#2a6f97"),
             ("cht_steel", "solid", "heater", "stainless plate 2 mm, bottom",
              "#d1495b")]
    for folder, region, patch, label, color in cases:
        x, t = patch_profile(work / "cht" / folder, region, patch)
        b.plot(x, t, color=color, lw=2, label=label)
        print(f"{label:<32} 최고 {t.max():.3f} K, 평균 {t.mean():.3f} K")
    b.set_xlabel("x (mm): heated section 10-70")
    b.set_ylabel("heated-surface temperature rise (K)")
    b.set_title("2D channel with a cooling plate (CHT)")
    b.grid(alpha=0.3)
    b.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(work / "a4-3d-cht.png", dpi=130)
    print("저장:", work / "a4-3d-cht.png")
