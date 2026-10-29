"""책 08-3·08-4 공통: 저장한 PhysicsNeMo FNO를 불러온다."""
import json
import sys
from pathlib import Path

import numpy as np
import physicsnemo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch07"))
from fno_data import FieldScaler  # noqa: E402


def load_pnfno(path):
    """train_pnfno.py가 저장한 .mdlus와 .json으로 (모델, 정규화, 설정)."""
    path = Path(path).expanduser()
    meta = json.loads(path.with_suffix(".json").read_text())
    # .mdlus에 생성 인자가 들어 있어 모델 클래스와 설정을 따로 적지 않아도 된다
    model = physicsnemo.Module.from_checkpoint(str(path.with_suffix(".mdlus")))
    scaler = FieldScaler()
    scaler.rms = np.array(meta["rms"], np.float32)
    return model.cuda().eval(), scaler, meta
