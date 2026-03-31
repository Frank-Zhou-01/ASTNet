import sys
import time

sys.path.append('/root/JIIFNet-main/')

import torch
from basicsr.archs.ASSTNet_arch import ASSTNet_arch


cuda = torch.device('cuda')
model = ASSTNet_arch(channels=64, transformer_list=[4, 6], head_list=[4, 8], win_size=[8, 8], fusion_blocks=15, refinement_blocks=5)
model = model.to(cuda)
model.eval()

x = torch.randn((1, 10, 3, 1280, 768))
x = x.to(cuda)
start = time.time()
with torch.no_grad():
    for i in range(50):
        torch.cuda.empty_cache()
        _ = model(x)
    end = time.time()
    speed = end - start
    print(speed / 500)


