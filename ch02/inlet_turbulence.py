U = 10.0      # 입구 속도 [m/s]
I = 0.05      # 난류 강도 5%
L = 0.00254   # 난류 길이 척도 [m], 입구 높이 25.4mm의 10%
Cmu = 0.09    # k-epsilon 모델 상수
nu = 1e-05    # 동점성계수 [m2/s]

k = 1.5 * (U * I) ** 2
epsilon = Cmu**0.75 * k**1.5 / L
Re = U * 0.0254 / nu
print(f"k = {k:.3f}, epsilon = {epsilon:.3f}, Re = {Re:.0f}")
