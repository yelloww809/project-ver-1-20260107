import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import stft
import os
import glob

# ==========================================
# 1. 人工设置参数 (Configuration)
# ==========================================

# 路径设置
DATASET_PATH = '/data/hwh_data_folder/dataset/train'
OUTPUT_DIR = './stft_results_HD' # 建议换个文件夹存高清图

FILE_INDEX = 10
FREQ_RES_KHZ = 5      # 极高分辨率测试 (建议设小一点，如 5 或 10，以此测试高清大图)
OVERLAP_RATIO = 0.5   

# --- 图像显示设置 ---
USE_GRAYSCALE = True  
DRAW_LABELS = True    
INVERT_FREQ_AXIS = False 

# --- 新增：尺寸控制策略 ---
# True  = 强制固定大小 (比如 12x8 英寸)。
#         缺点：高分辨率STFT会被压缩/插值，丢失窄带细节。
#         场景：制作CNN训练集。
# False = 自适应原始大小 (Native Resolution)。
#         优点：STFT矩阵有多少点，图就有多少像素，保证 100% 细节不丢失。
#         场景：人工检查窄带信号。注意：生成的图片可能非常巨大！
USE_FIXED_SIZE = False

# 如果不使用固定大小，定义绘图的 DPI (每英寸像素数)
# 也就是：figsize_inch = data_points / DPI
# 96 是常见的屏幕 DPI，你可以设为 72 或 100
NATIVE_DPI = 100 

# ==========================================
# 2. 核心处理函数
# ==========================================

def process_single_file(file_idx, dataset_dir, output_dir):
    bin_path = os.path.join(dataset_dir, f'{file_idx}.bin')
    json_path = os.path.join(dataset_dir, f'{file_idx}.json')

    if not os.path.exists(bin_path) or not os.path.exists(json_path):
        print(f"[Error] File not found: {file_idx}")
        return

    # --- Step 1: 读取及参数计算 ---
    with open(json_path, 'r') as f:
        label_data = json.load(f)

    obs_range = label_data['observation_range']
    f_min, f_max = obs_range[0], obs_range[1]
    
    bandwidth_mhz = f_max - f_min
    fs = bandwidth_mhz * 1e6 
    center_freq_mhz = (f_min + f_max) / 2.0 

    # --- Step 2: 信号读取 ---
    raw_data = np.fromfile(bin_path, dtype=np.float16)
    signal = raw_data[::2] + 1j * raw_data[1::2]

    # --- Step 3: STFT ---
    target_res_hz = FREQ_RES_KHZ * 1000.0
    nperseg = int(fs / target_res_hz)
    noverlap = int(nperseg * OVERLAP_RATIO)
    
    f_stft, t_stft, Zxx = stft(signal, fs=fs, window='hann', nperseg=nperseg, 
                               noverlap=noverlap, return_onesided=False)

    # --- Step 4: 坐标处理 ---
    Zxx = np.fft.fftshift(Zxx, axes=0)
    f_stft = np.fft.fftshift(f_stft)

    f_mhz = (f_stft / 1e6) + center_freq_mhz
    t_ms = t_stft * 1000.0
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-12)

    # --- Step 5: 动态决定画布大小 ---
    # 获取 STFT 矩阵的形状 (频点数, 时间步数)
    n_freq_bins, n_time_steps = Zxx.shape
    
    if USE_FIXED_SIZE:
        # 以前的逻辑：固定大小
        fig_w, fig_h = 12, 8
        my_dpi = 150
        print(f"Mode: Fixed Size (Canvas: {fig_w}x{fig_h} inch)")
    else:
        # 新逻辑：根据数据量决定画布大小
        # 加上一些边距给坐标轴 (比如宽高各加 2 英寸)
        # 注意：如果数据量极大，Matplotlib 可能会报错或内存溢出，但在 5kHz 分辨率下通常没事
        fig_w = (n_time_steps / NATIVE_DPI) + 2 
        fig_h = (n_freq_bins / NATIVE_DPI) + 2
        
        # 限制最小尺寸，防止文件太短导致图太小
        fig_w = max(fig_w, 8)
        fig_h = max(fig_h, 6)
        
        my_dpi = NATIVE_DPI
        print(f"Mode: Native Size (Data: {n_time_steps}x{n_freq_bins} -> Canvas: {fig_w:.1f}x{fig_h:.1f} inch)")

    plt.figure(figsize=(fig_w, fig_h))

    # --- Step 6: 绘图 ---
    if USE_GRAYSCALE:
        cmap_choice = 'gray' 
        box_color = 'white' if not INVERT_FREQ_AXIS else 'red' # 翻转时有时候换个颜色醒目点
        text_bg_color = 'black'
    else:
        cmap_choice = 'jet'
        box_color = 'white' 
        text_bg_color = 'black'

    # pcolormesh 是矢量绘制，shading='nearest' 能更好保留像素原本的方块感，不进行插值模糊
    # shading='auto' 或 'gouraud' 会平滑，不利于看噪点
    shading_mode = 'nearest' if not USE_FIXED_SIZE else 'auto'
    
    plt.pcolormesh(t_ms, f_mhz, magnitude_db, shading=shading_mode, cmap=cmap_choice)
    
    if DRAW_LABELS:
        cbar = plt.colorbar()
        cbar.set_label('Magnitude (dB)')

    ax = plt.gca()

    if INVERT_FREQ_AXIS:
        ax.invert_yaxis()

    # --- 绘制标签 ---
    if DRAW_LABELS:
        signals = label_data.get('signals', [])
        for sig in signals:
            cls_id = sig['class']
            start_f = sig['start_frequency']
            end_f = sig['end_frequency']
            start_t = sig['start_time']
            end_t = sig['end_time']

            width = end_t - start_t
            height = end_f - start_f
            
            # 统一使用白色框
            rect = patches.Rectangle((start_t, start_f), width, height, 
                                     linewidth=2 if USE_FIXED_SIZE else 1, # 大图时线条可以细一点
                                     edgecolor='white', 
                                     facecolor='none', linestyle='--')
            ax.add_patch(rect)
            
            text_y = end_f if not INVERT_FREQ_AXIS else start_f
            va = 'bottom' if not INVERT_FREQ_AXIS else 'top'
            
            # 大图时字体稍微大一点
            font_sz = 10 if USE_FIXED_SIZE else 12
            
            plt.text(start_t, text_y, f'Class {cls_id}', color='white', 
                     fontsize=font_sz, fontweight='bold', verticalalignment=va, 
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    axis_status = "Inverted" if INVERT_FREQ_AXIS else "Standard"
    plt.title(f'STFT - File: {file_idx}.bin\nRes: {FREQ_RES_KHZ}kHz | Matrix: {n_freq_bins}x{n_time_steps}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (MHz)')
    plt.ylim(f_min, f_max)
    
    if INVERT_FREQ_AXIS:
        ax.invert_yaxis()

    # 保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    label_suffix = "_labeled" if DRAW_LABELS else "_clean"
    size_suffix = "_fixed" if USE_FIXED_SIZE else "_native"
    
    save_path = os.path.join(output_dir, f'stft_{file_idx}{label_suffix}{size_suffix}.png')
    
    plt.savefig(save_path, dpi=my_dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==========================================
# 3. 主程序入口
# ==========================================

if __name__ == "__main__":
    if FILE_INDEX == -1:
        bin_files = glob.glob(os.path.join(DATASET_PATH, '*.bin'))
        indices = [int(os.path.basename(f).split('.')[0]) for f in bin_files]
        indices.sort()
        for idx in indices:
            process_single_file(idx, DATASET_PATH, OUTPUT_DIR)
    else:
        process_single_file(FILE_INDEX, DATASET_PATH, OUTPUT_DIR)