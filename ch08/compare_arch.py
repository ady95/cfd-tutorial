"""책 08-3: neuraloperator FNO와 PhysicsNeMo FNO의 구조를 나란히 본다 (CPU).

7장 설정(모드 16 x 64, 폭 32, 층 4개, 패딩 10%)과 같은 크기가 되도록
PhysicsNeMo 쪽 인자를 정하고, 부분별 파라미터 수와 패딩 뒤 크기를 비교한다.

사용법: uv run python ch08/compare_arch.py
"""
import torch
from neuralop.layers.padding import DomainPadding
from neuralop.models import FNO as NeuralopFNO
from physicsnemo.models.fno import FNO as PhysicsNeMoFNO


def count(m):
    """파라미터 수. 복소수 가중치는 실수 2개로 센다."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())


a = NeuralopFNO(n_modes=(16, 64), in_channels=2, out_channels=4,
                hidden_channels=32, n_layers=4, domain_padding=0.1)
b = PhysicsNeMoFNO(in_channels=2, out_channels=4, latent_channels=32,
                   num_fno_layers=4, num_fno_modes=[8, 33], padding=[10, 204],
                   decoder_layers=1, decoder_layer_size=32)

x = torch.zeros(1, 32, 49, 1025)
print("neuraloperator 패딩 뒤 크기:", tuple(DomainPadding(0.1).pad(x).shape[2:]))
seen = []                    # 첫 푸리에 층에 들어가는 텐서의 크기를 엿본다
b.spec_encoder.spconv_layers[0].register_forward_hook(
    lambda m, inp, out: seen.append(tuple(inp[0].shape[2:])))
with torch.no_grad():
    b(torch.zeros(1, 2, 49, 1025))
print("PhysicsNeMo 패딩 뒤 크기:  ", seen[0])

print("\nneuraloperator")
print(f"  입력 올리기 (lifting)       {count(a.lifting):>10,}")
print(f"  푸리에 층 4개 (fno_blocks)  {count(a.fno_blocks):>10,}")
spec = sum(count(c) for c in a.fno_blocks.convs)
print(f"    그중 스펙트럼 가중치      {spec:>10,}")
print(f"  출력 내리기 (projection)    {count(a.projection):>10,}")
print(f"  합계                        {count(a):>10,}")

enc = b.spec_encoder
print("\nPhysicsNeMo")
print(f"  입력 올리기 (lift_network)  {count(enc.lift_network):>10,}")
layers = count(enc.spconv_layers) + count(enc.conv_layers)
print(f"  푸리에 층 4개               {layers:>10,}")
print(f"    그중 스펙트럼 가중치      {count(enc.spconv_layers):>10,}")
print(f"  출력 내리기 (decoder_net)   {count(b.decoder_net):>10,}")
print(f"  합계                        {count(b):>10,}")
