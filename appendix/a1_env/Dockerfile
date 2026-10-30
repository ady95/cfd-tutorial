# 책 부록 A-1: 이 책의 실습 환경을 담은 Docker 이미지
#   Ubuntu 22.04 + OpenFOAM 14 (Foundation apt 패키지 20261008)
#   + uv 0.13.1 + Python 3.12 + 저장소 uv.lock의 패키지 전체
#
# 빌드 (저장소 루트에서):
#   docker build -f appendix/a1_env/Dockerfile -t cfd-tutorial:book .
# 실행 (저장소를 /work로 연결):
#   docker run --rm --gpus all -v $PWD:/work cfd-tutorial:book \
#       bash appendix/a1_env/ref_check.sh
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 TZ=Asia/Seoul
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget git bc \
 && wget -qO /etc/apt/trusted.gpg.d/openfoam.asc https://dl.openfoam.org/gpg.key \
 && echo "deb http://dl.openfoam.org/ubuntu jammy main" \
      > /etc/apt/sources.list.d/openfoam.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends openfoam14=20261008 \
 && apt-mark hold openfoam14 \
 && rm -rf /var/lib/apt/lists/*

# uv는 버전을 고정한 공식 이미지에서 실행 파일만 가져온다
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# 패키지는 잠금 파일 그대로 설치한다. 가상환경은 /opt/venv에 두어,
# 실행할 때 연결하는 저장소의 .venv(호스트용)와 섞이지 않게 한다
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache
WORKDIR /opt/lock
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-install-project && rm -rf /tmp/uv-cache \
 && chmod -R a+rX /opt/venv /opt/python

# 연결한 저장소에 __pycache__를 남기지 않는다
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
WORKDIR /work
RUN echo ". /opt/openfoam14/etc/bashrc" >> /etc/bash.bashrc
CMD ["bash"]
