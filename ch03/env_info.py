"""책 03-5: 해석 결과를 다시 만들 때 필요한 환경 정보를 모은다.

사용법: uv run python ch03/env_info.py
다른 스크립트에서는 env_info()가 dict를 돌려준다. record_env(작업 폴더)는
작업 폴더에 env.json으로 저장한다 (run_cases.py, ch04/run_batch.py).
호스트 이름이나 사용자 이름처럼 환경을 다시 만드는 데 필요 없는 정보는
담지 않는다.
"""
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git(*args):
    p = subprocess.run(["git", "-C", str(REPO), *args],
                       capture_output=True, text=True)
    return p.stdout.strip()


def env_info():
    foam = Path(os.environ.get("WM_PROJECT_DIR", ""))
    build = foam / ".build"
    os_name = ""
    for line in Path("/etc/os-release").read_text().splitlines():
        if line.startswith("PRETTY_NAME="):
            os_name = line.split("=", 1)[1].strip('"')
    cpu = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "os": os_name,
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "openfoam": build.read_text().strip() if build.exists() else None,
        "python": platform.python_version(),
        "repo_commit": git("rev-parse", "--short", "HEAD"),
        # 커밋하지 않은 수정이 있으면 그 커밋만으로는 결과를 다시 만들 수 없다
        "repo_modified": git("status", "--porcelain") != "",
    }


def record_env(run_dir):
    """작업 폴더에 env.json을 남긴다. 이미 있으면 바뀐 항목 이름을 돌려준다."""
    env, path = env_info(), Path(run_dir) / "env.json"
    if not path.exists():
        path.write_text(json.dumps(env, ensure_ascii=False, indent=2))
        return []
    old = json.loads(path.read_text())
    return [k for k in ("openfoam", "repo_commit", "repo_modified")
            if old.get(k) != env[k]]


if __name__ == "__main__":
    print(json.dumps(env_info(), ensure_ascii=False, indent=2))
