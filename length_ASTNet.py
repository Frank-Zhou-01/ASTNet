import matplotlib.pyplot as plt
import numpy as np

# 数据
x = [5, 10, 15, 20, 25, 30]
y = [34.05, 34.24, 34.31, 34.35, 34.38, 34.38]

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 创建图形
plt.figure(figsize=(20, 7))

# 绘制折线图
plt.plot(x, y, marker='o', markersize=15, linewidth=3, color='#4F9D9D',
         markerfacecolor='#336666', markeredgewidth=1, markeredgecolor='black')

# 设置坐标轴标签
plt.xlabel('Video Length (frame)', fontsize=40, labelpad=10)
plt.ylabel('PSNR (dB)', fontsize=40, labelpad=10)

# 设置坐标轴范围
# plt.xlim(0, 32)
# plt.ylim(34.0, 34.7)

# 设置坐标轴边框粗细
ax = plt.gca()  # 获取当前坐标轴
ax.spines['top'].set_linewidth(2.0)      # 上边框
ax.spines['right'].set_linewidth(2.0)    # 右边框
ax.spines['bottom'].set_linewidth(2.0)   # 下边框
ax.spines['left'].set_linewidth(2.0)     # 左边框

# 设置刻度
plt.xticks(x, fontsize=35)
plt.yticks(np.arange(34.0, 34.4, 0.1), fontsize=35)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 调整边框
for spine in plt.gca().spines.values():
    spine.set_linewidth(2.0)

# 显示图形
plt.tight_layout()
plt.show()