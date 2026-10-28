"""책 02-4: ParaView 스크립트(pvbatch)로 pitzDailySteady 속도장 그림을 만든다.

사용법 (Ubuntu 22.04 apt 패키지 ParaView 5.10 기준):
    export PYTHONPATH=/usr/lib/python3/dist-packages
    xvfb-run -a pvbatch ch02/pv_pitzdaily.py ~/cfd-runs/pitzDailySteady
결과: 케이스 폴더의 images/pitzDaily-U.png
"""
import sys
from pathlib import Path

import paraview.simple as pvs

case = Path(sys.argv[1]).expanduser()
foam = case / "case.foam"
foam.touch()    # OpenFOAM 리더가 케이스를 알아보는 빈 파일
out = case / "images"
out.mkdir(exist_ok=True)

reader = pvs.OpenFOAMReader(FileName=str(foam))
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = ["U", "p"]
reader.UpdatePipelineInformation()
last = reader.TimestepValues[-1]
print("ParaView", pvs.GetParaViewVersion(), "/ 마지막 반복", last)

view = pvs.GetActiveViewOrCreate("RenderView")
view.ViewSize = [1600, 420]
view.ViewTime = last
view.OrientationAxesVisibility = 0
view.InteractionMode = "2D"
view.UseColorPaletteForBackground = 0   # 책에 싣기 좋게 흰 배경
view.Background = [1, 1, 1]

display = pvs.Show(reader, view)
pvs.ColorBy(display, ("POINTS", "U", "Magnitude"))
display.RescaleTransferFunctionToDataRange(True, False)
display.SetScalarBarVisibility(view, True)
bar = pvs.GetScalarBar(pvs.GetColorTransferFunction("U"), view)
bar.Title = "U magnitude [m/s]"
bar.ComponentTitle = ""
bar.Orientation = "Horizontal"
bar.WindowLocation = "Lower Center"
bar.TitleColor = [0, 0, 0]
bar.LabelColor = [0, 0, 0]
bar.RangeLabelFormat = "%.0f"

view.ResetCamera()
pvs.GetActiveCamera().Zoom(3.2)     # 화면 폭에 맞게 확대
pvs.SaveScreenshot(str(out / "pitzDaily-U.png"), view,
                   ImageResolution=[1600, 420])
print("저장:", out / "pitzDaily-U.png")
