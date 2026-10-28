# 3장. 자동차 냉각 채널 해석 모델 만들기

OpenFOAM 14와 bash 환경이 필요합니다(책 02-1). GPU는 쓰지 않습니다.

아래 벽(가열면)은 평평하고 위 벽(덮개)에 사각 리브 N개가 매달린
2D 냉각 채널입니다.
설계 변수는 채널 높이 H, 리브 높이 e, 리브 개수 N, 입구 유속 U입니다.

| 파일 | 책 | 내용 |
|---|---|---|
| template/ | 03-1 | 케이스 원본 (blockMeshDict 제외). 온도·열유속 설정 포함 |
| channel_mesh.py | 03-1 | H, e, N으로 blockMeshDict 생성 |
| mesh_check.sh | 03-1 | 변수 범위 양 끝 형상 5개의 격자 생성·검사 |
| plot_mesh.py | 03-1 | 첫 번째 리브 부근 격자 그림 |

```bash
cp -r ch03/template ~/cfd-runs/rib-base
uv run python ch03/channel_mesh.py 2 0.6 8 \
    ~/cfd-runs/rib-base/system/blockMeshDict

bash ch03/mesh_check.sh                     # ~/cfd-runs/rib-check
uv run python ch03/plot_mesh.py ~/cfd-runs/rib-check/H2_e0.6_N8
```
