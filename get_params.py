import sys
sys.path.append('/root/REAL-Methods/ASTNet-main/')

from basicsr.archs.ASTNetL_arch import ASTNetL_arch
import torch
from thop import profile


def get_params(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("总参数数量和: " + f'{(k / (1024 * 1024)):.2f}M')


def count_flops(name, model):
    device = torch.device('cuda')
    model = model.to(device)
    x = torch.randn((1, 1, 3, 256, 256)).to(device)

    flops, params = profile(model, inputs=(x,))

    print(f'{name}: FLOPS: {flops / (1024 * 1024 * 1024)}G, Params: {params / (1024 * 1024)}M')


net = ASTNetL_arch(channels=64, transformer_list=[5, 6], head_list=[4, 8], fusion_blocks=16, refinement_blocks=5, win_size=[8, 8])

count_flops('ASTNet', net)


