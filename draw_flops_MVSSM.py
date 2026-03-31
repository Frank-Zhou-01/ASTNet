import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm

# 查看所有支持中文的字体
chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'Microsoft' in f.name or 'Kai' in f.name or 'Song' in f.name]
print(chinese_fonts)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 数据
FLOPS = {
    'STFAN': 175.2,
    'EDVR': 194.2,
    'CDVDTSP': 357.9,
    'MPRNet': 760.1,
    'Restormer': 141.0,
    'RNN-MBP': 496.0,
    'VRT': 721.3,
    'MVSSM': 64.7,
    'MVSSM-L': 186.9,
    'FGST': 131.6,
    'Turtle': 178.8,
    'STDAN': 213.1,
    'CDVDTSPNL': 484.1,
    'MambaIR': 439.0,
    'EAMamba': 137,
    'PGDN': 613.0
}

PSNR = {
    'STFAN': 28.59,
    'EDVR': 26.83,
    'CDVDTSP': 31.67,
    'MPRNet': 32.73,
    'Restormer': 32.92,
    'RNN-MBP': 33.32,
    'VRT': 34.81,
    'MVSSM': 34.66,
    'MVSSM-L': 35.66,
    'FGST': 32.90,
    'Turtle': 34.50,
    'STDAN': 32.29,
    'CDVDTSPNL': 33.44,
    'MambaIR': 33.21,
    'EAMamba': 33.58,
    'PGDN': 34.17
}

manual_sizes = {
    'STFAN': 1400,
    'EDVR': 1800,
    'CDVDTSP': 2500,
    'MPRNet': 10000,
    'Restormer': 1200,
    'RNN-MBP': 4000,
    'FGST': 1100,
    'VRT': 7000,
    'MVSSM': 1000,
    'MVSSM-L': 1100,
    'Turtle': 1500,
    'STDAN': 2000,
    'CDVDTSPNL': 3500,
    'MambaIR': 3000,
    'EAMamba': 1050,
    'PGDN': 4500
}

# 创建图形
plt.figure(figsize=(20, 10))

# 提取数据
methods = list(FLOPS.keys())
flops_values = [FLOPS[method] for method in methods]
psnr_values = [PSNR[method] for method in methods]
sizes = [manual_sizes[method] for method in methods]

# 根据FLOPS值调整点的大小（归一化到合适的范围）
# sizes = [(flops / max(flops_values)) * 300 + 50 for flops in flops_values]  # 大小范围：50-350

# 绘制散点图
scatter = plt.scatter(flops_values, psnr_values, s=sizes, alpha=0.7,
                      c=flops_values, cmap='viridis', edgecolors='#ffffff')

# 添加颜色条并设置字体大小
# cbar = plt.colorbar(scatter)
# cbar.set_label('FLOPS (G)', rotation=270, labelpad=15, fontsize=35)  # 颜色条标签字体大小
# cbar.ax.tick_params(labelsize=15)  # 颜色条刻度字体大小

# 在每个点上标注方法名称
for i, method in enumerate(methods):

    xytext = (0, 0)
    ha = 'center'
    va = 'bottom'

    # 为特定方法设置特殊位置
    if method == 'FGST':
        xytext = (-40, -50)  # 向右上方偏移
        ha = 'center'
        va = 'bottom'

    if method == 'Restormer':
        xytext = (80, 5)  # 向右上方偏移
        ha = 'center'
        va = 'bottom'

    if method == 'Turtle':
        xytext = (50, 5)  # 向右上方偏移
        ha = 'center'
        va = 'bottom'

    if method == 'RNN-MBP':
        xytext = (90, -50)  # 向右上方偏移
        ha = 'center'
        va = 'bottom'

    if method == 'CDVDTSPNL':
        xytext = (-100, 20)  # 向右上方偏移
        ha = 'center'
        va = 'bottom'

    if method == 'MVSSM':
        xytext = (10, 10)

    if method == 'MambaIR':
        xytext = (0, -55)

    if method == 'EAMamba':
        xytext = (0, 10)

    if method == 'STDAN':
        xytext = (0, -50)

    if method == 'STFAN':
        xytext = (0, 20)

    if method == 'EDVR':
        xytext = (0, 20)

    if method == 'CDVDTSP':
        xytext = (0, -60)

    if method == 'VRT':
        xytext = (0, 40)

    if method == 'MPRNet':
        xytext = (-20, -80)

    if method == 'MVSSM-L':
        xytext = (-90, -10)

    if method == 'PGDN':
        xytext = (0, 30)

    plt.annotate(method,
                 (flops_values[i], psnr_values[i]),
                 xytext=xytext,  # 设置为(0,0)表示没有偏移
                 textcoords='offset points',
                 fontsize=30,
                 fontweight='bold',
                 alpha=0.8,
                 fontfamily='Times New Roman',
                 ha=ha,  # 水平居中
                 va=va)  # 垂直居中

# 设置坐标轴边框粗细
ax = plt.gca()  # 获取当前坐标轴
ax.spines['top'].set_linewidth(2.0)      # 上边框
ax.spines['right'].set_linewidth(2.0)    # 右边框
ax.spines['bottom'].set_linewidth(2.0)   # 下边框
ax.spines['left'].set_linewidth(2.0)     # 左边框

# 设置坐标轴标签和标题
plt.xlabel('FLOPs (G)', fontsize=45)
plt.ylabel('PSNR (dB)', fontsize=45)

plt.xticks(fontsize=35)
plt.yticks([26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], fontsize=35)

# 添加网格
plt.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
