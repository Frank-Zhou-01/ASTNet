import os


root = '/root/FastTurbNet/datasets/Mix_data/train/turb'

dirs = os.listdir(root)
print(dirs)
for d in dirs:
    turbs = os.listdir(os.path.join(root, d))
    turbs.sort(key=lambda x: int(x[:-4]))
    index = 0
    print(d)
    for i in turbs:
        os.rename(src=os.path.join(root, d, i), dst=os.path.join(root, d, f'{index:06d}.png'))
        index += 1
