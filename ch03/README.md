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
| plot_fields.py | 03-2 | 속도 크기·온도 분포 그림 (x = 8~40 mm) |
| condition_check.sh | 03-2 | 열유속·입구 온도·유속을 바꾼 해석 5개 비교 |
| extract.py | 03-3 | 압력손실·온도 상승 추출과 에너지 수지 계산 |
| extract_check.sh | 03-3 | 범위 끝 형상·유속 5개 해석과 에너지 수지 검증 |

```bash
cp -r ch03/template ~/cfd-runs/rib-base
uv run python ch03/channel_mesh.py 2 0.6 8 \
    ~/cfd-runs/rib-base/system/blockMeshDict

bash ch03/mesh_check.sh                     # ~/cfd-runs/rib-check
uv run python ch03/plot_mesh.py ~/cfd-runs/rib-check/H2_e0.6_N8

~/cfd-runs/rib-base/Allrun                  # 기준 형상 해석 (약 5초)
uv run python ch03/plot_fields.py ~/cfd-runs/rib-base
bash ch03/condition_check.sh               # ~/cfd-runs/cond
uv run python ch03/extract.py ~/cfd-runs/rib-base
bash ch03/extract_check.sh                 # ~/cfd-runs/extract
```
