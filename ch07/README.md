# 7장. FNO로 온도장·속도장 예측하기

4장에서 해석한 케이스 폴더의 물리장을 균일 격자 배열로 옮기고,
neuraloperator의 FNO로 온도장·속도장·압력장 전체를 예측합니다.
GPU가 필요합니다(이 책은 RTX 3090 24GB). 격자 변환에는 04-3의 케이스 폴더
~/cfd-runs/dataset(3.6GB)이 있어야 합니다.

| 파일 | 책 | 내용 |
|---|---|---|
| grid.py | 07-3 | OpenFOAM 결과 -> 균일 격자 (보간, 마스크, 가열면 온도 보정), 격자에서 dp·dT_max·dT_avg 다시 계산 |
| resolution.py | 07-3 | 격자 해상도 4가지의 옮기기 오차 비교 |
| build_fields.py | 07-3 | valid 690개를 1025 x 49로 옮겨 ~/cfd-runs/ch07/fields_1025x49.npz (0.59 GB) |
| fno_data.py | 07-3 | 입력 채널(마스크, 유속), 출력 정규화, 학습용 텐서 |
| show_sample.py | 07-3 | 케이스 하나의 입력·출력 채널 그림 |
| spectrum.py | 07-2 | 물리장을 푸리에 모드로 잘라 되돌린 오차 |
| fno_cost.py | 07-2 | 해상도·모드·폭별 파라미터 수, 학습 스텝 시간, GPU 메모리 |
| train_fno.py | 07-4 | FNO 학습, fno.pt(가중치)와 fno.json(설정·정규화) 저장 |
| fno_model.py | 07-4, 07-5 | 저장한 FNO 불러오기, 예측, 상대 L2 오차 |
| evaluate_fno.py | 07-5 | split별 물리장 오차, 물리장에서 다시 계산한 값과 MLP 비교, 추론 시간, 그림 |
| resolution_transfer.py | 07-5 | 학습 해상도와 다른 격자에서 실행 |

```bash
uv run python ch07/resolution.py            # 약 20초 (CPU 6코어)
uv run python ch07/build_fields.py          # 약 1분, 0.59 GB
uv run python ch07/show_sample.py H1.601_e0.329_N11_U0.0768
uv run python ch07/spectrum.py
uv run python ch07/fno_cost.py              # GPU가 비어 있을 때
uv run python ch07/train_fno.py             # 약 16분 (RTX 3090)
uv run python ch07/train_fno.py --step 2 --out ~/cfd-runs/ch07/fno_half
uv run python ch07/evaluate_fno.py
uv run python ch07/resolution_transfer.py
```

그림과 모델은 ~/cfd-runs/ch07에 저장됩니다. 학습한 FNO(약 17 MB)는 저장소에
넣지 않았습니다. 위 명령으로 다시 만들 수 있습니다.
