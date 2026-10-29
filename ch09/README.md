# 9장. 물리 법칙을 반영한 AI 모델 개발하기

7장 FNO의 손실에 질량·에너지 보존 손실을 더해 학습하고, 적용 전후의 정확도와
보존 오차를 비교합니다. 7장의 ~/cfd-runs/ch07/fields_1025x49.npz와 fno.pt,
8장의 ~/cfd-runs/ch08/extra/no_s1(7장 FNO 시드 1)이 있어야 합니다.
GPU가 필요합니다(이 책은 RTX 3090 24GB).

| 파일 | 책 | 내용 |
|---|---|---|
| physics.py | 9장 공통 | 단면 적분 가중치, 기대 유량·열 유량, 질량·에너지 잔차 (numpy, torch 모두) |
| conservation.py | 09-2 | CFD 격자 데이터와 7장 FNO의 보존 잔차, 경계조건, 흐름 방향 열확산 |
| train_pifno.py | 09-3 | 데이터 손실 + λ x 물리 손실로 FNO 학습 (가중치 올리기, 허용 오차, train 개수, 정답 없는 입력) |
| compare_pi.py | 09-4, 09-5 | 여러 모델의 split별 물리장 오차, 보존 잔차, 다시 계산한 세 값 |
| plot_ood.py | 09-4 | 범위 밖 높이 케이스 하나의 온도장과 단면 유량 |
| grad_balance.py | 09-5 | 손실 항별 파라미터 기울기 크기 |

```bash
uv run python ch09/conservation.py          # ~/cfd-runs/ch09/conservation.png
d=~/cfd-runs/ch09/full
uv run python ch09/train_pifno.py --lam-mass 0.1 --lam-energy 0.1 --out $d/lam01
uv run python ch09/train_pifno.py --out $d/lam1                  # λ 기본값 1
uv run python ch09/train_pifno.py --colloc 400 --colloc-range wide --out $d/lam1_wide
uv run python ch09/compare_pi.py
uv run python ch09/plot_ood.py              # ~/cfd-runs/ch09/ood-case.png
uv run python ch09/train_pifno.py --warmup 100 --out $d/lam1_wu
uv run python ch09/grad_balance.py
```

원래 해상도 학습은 혼자 돌리면 하나에 약 16분(RTX 3090)이 걸립니다. 정답 없는 입력을
쓰면 묶음마다 계산이 두 배라 더 걸립니다.

- --out 이름에 점(.)을 쓰지 않습니다. 확장자를 붙일 때 with_suffix를 써서 lam0.1이
  lam0.pt로 저장되기 때문입니다. 스크립트가 시작할 때 막습니다.
- 물리 손실을 쓰면 같은 시드로 다시 학습해도 중간 기록이 조금 다를 수 있습니다.
  마지막 결과는 0.01%p 단위에서만 달랐습니다(09-3).
