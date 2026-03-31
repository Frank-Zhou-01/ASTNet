import sys
import time

sys.path.append('/root/auto-tmp/ASTNet-main/')

from MPRNet import MPRNet
from basicsr.archs.ASSTNet_arch import ASSTNet_arch
import torch

cuda = torch.device('cuda:0')
model = ASSTNet_arch(channels=32, transformer_list=[4, 6], head_list=[4, 8], win_size=[8, 8], fusion_blocks=15, refinement_blocks=0)
# model = GShiftNet()

# model = MPRNet()
model = model.to(cuda)
model.eval()

x = torch.randn((1, 3, 3, 768, 1280), dtype=torch.float32).to(cuda)
with torch.no_grad():
    start = time.time()
    for i in range(50):
        torch.cuda.empty_cache()
        _ = model(x)
        print(i + 1)
    end = time.time()

speed = ((end - start) * 1000) / 150
print(speed)
