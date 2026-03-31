import numpy as np
import cv2
import matplotlib.pyplot as plt
from pywt import dwt2, idwt2
import os


def haar_wavelet_transform_rgb(image_path, output_dir='wavelet_output'):
    """
    使用Haar小波变换对RGB图像进行变换并保存子带

    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转换颜色空间 BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]

    # 分离RGB通道
    r_channel = image_rgb[:, :, 0].astype(np.float64)
    g_channel = image_rgb[:, :, 1].astype(np.float64)
    b_channel = image_rgb[:, :, 2].astype(np.float64)

    # 对每个通道进行Haar小波变换
    coeffs_r = dwt2(r_channel, 'haar')
    coeffs_g = dwt2(g_channel, 'haar')
    coeffs_b = dwt2(b_channel, 'haar')

    # 获取小波系数的尺寸
    ll_height, ll_width = coeffs_r[0].shape
    detail_height, detail_width = coeffs_r[1][0].shape

    print(f"原始图像尺寸: {original_height}x{original_width}")
    print(f"近似系数尺寸: {ll_height}x{ll_width}")
    print(f"细节系数尺寸: {detail_height}x{detail_width}")

    # 创建正确尺寸的子带图像
    subbands = {}

    # 近似子带 (LL) - 尺寸减半
    ll_band = np.zeros((ll_height, ll_width, 3), dtype=np.float64)
    ll_band[:, :, 0] = coeffs_r[0]  # R通道
    ll_band[:, :, 1] = coeffs_g[0]  # G通道
    ll_band[:, :, 2] = coeffs_b[0]  # B通道

    # 水平细节子带 (HL) - 尺寸减半
    hl_band = np.zeros((detail_height, detail_width, 3), dtype=np.float64)
    hl_band[:, :, 0] = coeffs_r[1][0]  # R通道水平细节
    hl_band[:, :, 1] = coeffs_g[1][0]  # G通道水平细节
    hl_band[:, :, 2] = coeffs_b[1][0]  # B通道水平细节

    # 垂直细节子带 (LH) - 尺寸减半
    lh_band = np.zeros((detail_height, detail_width, 3), dtype=np.float64)
    lh_band[:, :, 0] = coeffs_r[1][1]  # R通道垂直细节
    lh_band[:, :, 1] = coeffs_g[1][1]  # G通道垂直细节
    lh_band[:, :, 2] = coeffs_b[1][1]  # B通道垂直细节

    # 对角细节子带 (HH) - 尺寸减半
    hh_band = np.zeros((detail_height, detail_width, 3), dtype=np.float64)
    hh_band[:, :, 0] = coeffs_r[1][2]  # R通道对角细节
    hh_band[:, :, 1] = coeffs_g[1][2]  # G通道对角细节
    hh_band[:, :, 2] = coeffs_b[1][2]  # B通道对角细节

    subbands = {
        'LL': ll_band,
        'HL': hl_band,
        'LH': lh_band,
        'HH': hh_band
    }

    # 保存子带图像 - 针对不同子带使用不同的归一化策略
    for name, band in subbands.items():
        if name == 'LL':
            # 对于低频子带，直接归一化到0-255
            band_normalized = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)
        else:
            # 对于高频子带，使用增强的归一化来保留颜色信息
            band_normalized = enhance_high_frequency_bands(band)

        band_uint8 = band_normalized.astype(np.uint8)

        # 保存为RGB图像
        output_path = os.path.join(output_dir, f'{name}_subband.png')
        cv2.imwrite(output_path, cv2.cvtColor(band_uint8, cv2.COLOR_RGB2BGR))
        print(f"保存: {output_path} - 尺寸: {band_uint8.shape[1]}x{band_uint8.shape[0]}")

    # 创建完整的子带可视化（将所有子带组合成一个图像）
    create_composite_visualization(subbands, original_height, original_width, output_dir)

    # 可视化结果
    visualize_results(image_rgb, subbands, output_dir)

    return subbands


def enhance_high_frequency_bands(band):
    """
    增强高频子带的颜色信息
    """
    # 方法1: 使用标准差归一化来增强细节
    enhanced_band = np.zeros_like(band)

    for i in range(3):  # 对每个通道分别处理
        channel = band[:, :, i]

        # 计算均值和标准差
        mean_val = np.mean(channel)
        std_val = np.std(channel)

        if std_val > 0:
            # 使用均值和标准差进行归一化，增强对比度
            enhanced_channel = (channel - mean_val) / (3 * std_val) + 0.5
            enhanced_channel = np.clip(enhanced_channel * 255, 0, 255)
        else:
            enhanced_channel = np.zeros_like(channel)

        enhanced_band[:, :, i] = enhanced_channel

    return enhanced_band


def normalize_with_color_preservation(band):
    """
    另一种保留颜色的归一化方法
    """
    # 分别对每个通道进行归一化
    normalized_band = np.zeros_like(band)

    for i in range(3):
        channel = band[:, :, i]
        min_val = np.min(channel)
        max_val = np.max(channel)

        if max_val > min_val:
            normalized_channel = (channel - min_val) / (max_val - min_val) * 255
        else:
            normalized_channel = np.zeros_like(channel)

        normalized_band[:, :, i] = normalized_channel

    return normalized_band


def create_composite_visualization(subbands, orig_h, orig_w, output_dir):
    """
    创建完整的子带可视化图像，将所有子带组合成一个图像
    """
    # 获取各个子带的尺寸
    ll_h, ll_w = subbands['LL'].shape[:2]
    detail_h, detail_w = subbands['HL'].shape[:2]

    # 创建组合图像（尺寸与原始图像相同）
    composite = np.zeros((orig_h, orig_w, 3), dtype=np.float64)

    # 对高频子带进行增强处理
    ll_enhanced = cv2.normalize(subbands['LL'], None, 0, 255, cv2.NORM_MINMAX)
    hl_enhanced = enhance_high_frequency_bands(subbands['HL'])
    lh_enhanced = enhance_high_frequency_bands(subbands['LH'])
    hh_enhanced = enhance_high_frequency_bands(subbands['HH'])

    # 放置各个子带（需要上采样到原始尺寸）
    # LL 子带放在左上角
    composite[:ll_h, :ll_w] = cv2.resize(ll_enhanced, (ll_w, ll_h))

    # HL 子带放在右上角
    composite[:detail_h, ll_w:ll_w + detail_w] = cv2.resize(hl_enhanced, (detail_w, detail_h))

    # LH 子带放在左下角
    composite[ll_h:ll_h + detail_h, :detail_w] = cv2.resize(lh_enhanced, (detail_w, detail_h))

    # HH 子带放在右下角
    composite[ll_h:ll_h + detail_h, ll_w:ll_w + detail_w] = cv2.resize(hh_enhanced, (detail_w, detail_h))

    # 归一化并保存组合图像
    composite_normalized = cv2.normalize(composite, None, 0, 255, cv2.NORM_MINMAX)
    composite_uint8 = composite_normalized.astype(np.uint8)

    output_path = os.path.join(output_dir, 'composite_subbands.png')
    cv2.imwrite(output_path, cv2.cvtColor(composite_uint8, cv2.COLOR_RGB2BGR))
    print(f"保存组合图像: {output_path}")


def visualize_results(original, subbands, output_dir):
    """
    可视化原始图像和小波变换结果
    """
    plt.figure(figsize=(15, 10))

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title('原始图像')
    plt.axis('off')

    # 各子带图像
    bands = ['LL', 'HL', 'LH', 'HH']
    titles = ['近似系数 (LL)', '水平细节 (HL)', '垂直细节 (LH)', '对角细节 (HH)']

    for i, (band, title) in enumerate(zip(bands, titles)):
        plt.subplot(2, 3, i + 2)
        band_data = subbands[band]

        if band == 'LL':
            band_normalized = cv2.normalize(band_data, None, 0, 255, cv2.NORM_MINMAX)
        else:
            band_normalized = enhance_high_frequency_bands(band_data)

        band_uint8 = band_normalized.astype(np.uint8)
        plt.imshow(band_uint8)
        plt.title(f'{title}\n{band_uint8.shape[1]}x{band_uint8.shape[0]}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wavelet_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()


def reconstruct_image_from_subbands(subbands):
    """
    从小波子带重建图像
    """
    # 对每个通道进行逆小波变换
    reconstructed_channels = []

    for i in range(3):  # R, G, B通道
        # 提取各子带
        cA = subbands['LL'][:, :, i]
        cH = subbands['HL'][:, :, i]
        cV = subbands['LH'][:, :, i]
        cD = subbands['HH'][:, :, i]

        # 逆小波变换
        coeffs = (cA, (cH, cV, cD))
        reconstructed_channel = idwt2(coeffs, 'haar')
        reconstructed_channels.append(reconstructed_channel)

    # 合并通道
    reconstructed = np.stack(reconstructed_channels, axis=2)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    return reconstructed


# 使用示例
if __name__ == "__main__":
    # 示例用法
    image_path = "0001.png"  # 请替换为您的图像路径

    try:
        # 进行小波变换
        subbands = haar_wavelet_transform_rgb(image_path)

        # 重建图像
        reconstructed = reconstruct_image_from_subbands(subbands)

        # 保存重建图像
        cv2.imwrite('wavelet_output/reconstructed_image.png',
                    cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))
        print("重建图像已保存: wavelet_output/reconstructed_image.png")

        # 显示重建结果对比
        original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title('原始图像')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed)
        plt.title('重建图像')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('wavelet_output/comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"错误: {e}")
        print("请确保:")
        print("1. 已安装所需库: pip install opencv-python matplotlib PyWavelets numpy")
        print("2. 图像路径正确")
        print("3. 图像文件存在且可读")