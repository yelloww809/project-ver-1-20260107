import numpy as np
import json
import os
import glob
from scipy.signal import stft
from tqdm import tqdm

# ==========================================
# 1. 配置参数 (必须与预处理代码保持一致)
# ==========================================

DATASET_DIR = '/data/hwh_data_folder/dataset/train'  # 数据集路径
OUTPUT_LOG_FILE = 'dataset_db_stats.txt'             # 统计结果保存路径

# STFT 参数
# [重要] 这里的频率分辨率必须和你做数据集预处理时的分辨率一致
# 否则计算出的幅度值可能会有偏差 (因为能量分布受窗长影响)
FREQ_RES_KHZ = 5     # 5 kHz
OVERLAP_RATIO = 0.5   # 50%

# ==========================================
# 2. 核心统计逻辑
# ==========================================

def calculate_dataset_stats():
    # 1. 扫描文件
    if not os.path.exists(DATASET_DIR):
        print(f"Error: 路径不存在 {DATASET_DIR}")
        return

    bin_files = glob.glob(os.path.join(DATASET_DIR, '*.bin'))
    # 按数字序号排序，保证 log 也是有序的
    indices = sorted([int(os.path.basename(f).split('.')[0]) for f in bin_files])
    
    print(f"找到 {len(indices)} 个文件，开始统计 dB 范围...")
    print(f"STFT 分辨率: {FREQ_RES_KHZ} kHz")

    # 初始化全局极值
    global_min_db = float('inf')
    global_max_db = float('-inf')

    # 打开文件准备写入 (使用 'w' 模式，实时写入)
    with open(OUTPUT_LOG_FILE, 'w', encoding='utf-8') as f_log:
        # 写入表头
        f_log.write(f"Dataset Intensity Statistics Report\n")
        f_log.write(f"Source: {DATASET_DIR}\n")
        f_log.write(f"Config: FreqRes={FREQ_RES_KHZ}kHz, Overlap={OVERLAP_RATIO}\n")
        f_log.write("-" * 50 + "\n")
        f_log.write(f"{'File_ID':<10} | {'Min_dB':<15} | {'Max_dB':<15}\n")
        f_log.write("-" * 50 + "\n")

        # 使用 tqdm 显示进度条
        for idx in tqdm(indices, desc="Computing"):
            bin_path = os.path.join(DATASET_DIR, f'{idx}.bin')
            json_path = os.path.join(DATASET_DIR, f'{idx}.json')

            try:
                # --- A. 读取参数 ---
                with open(json_path, 'r') as f_json:
                    meta = json.load(f_json)
                
                obs_range = meta['observation_range']
                bw_mhz = obs_range[1] - obs_range[0]
                fs = bw_mhz * 1e6  # 采样率

                # --- B. 读取信号 ---
                raw = np.fromfile(bin_path, dtype=np.float16)
                signal = raw[::2] + 1j * raw[1::2]

                # --- C. STFT 处理 ---
                target_res_hz = FREQ_RES_KHZ * 1000.0
                nperseg = int(fs / target_res_hz)
                noverlap = int(nperseg * OVERLAP_RATIO)
                
                # 注意：return_onesided=False 
                _, _, Zxx = stft(signal, fs=fs, window='hann', nperseg=nperseg, 
                                 noverlap=noverlap, return_onesided=False)

                # --- D. 转 dB ---
                # 加 1e-12 防止 log(0)
                magnitude = np.abs(Zxx)
                db_values = 20 * np.log10(magnitude + 1e-12)

                # --- E. 统计当前文件 ---
                local_min = np.min(db_values)
                local_max = np.max(db_values)

                # 更新全局统计
                if local_min < global_min_db:
                    global_min_db = local_min
                if local_max > global_max_db:
                    global_max_db = local_max

                # 写入日志 (保留2位小数)
                f_log.write(f"{idx:<10} | {local_min:<15.2f} | {local_max:<15.2f}\n")

            except Exception as e:
                print(f"Error processing {idx}: {e}")
                f_log.write(f"{idx:<10} | ERROR           | {str(e)}\n")

        # --- F. 最终汇总 ---
        f_log.write("-" * 50 + "\n")
        f_log.write("FINAL RESULT (GLOBAL STATISTICS)\n")
        f_log.write(f"Global Min dB: {global_min_db:.4f}\n")
        f_log.write(f"Global Max dB: {global_max_db:.4f}\n")
        f_log.write("-" * 50 + "\n")

    print(f"\n统计完成！")
    print(f"全局最小值: {global_min_db:.2f} dB")
    print(f"全局最大值: {global_max_db:.2f} dB")
    print(f"详细报告已保存至: {os.path.abspath(OUTPUT_LOG_FILE)}")

if __name__ == "__main__":
    calculate_dataset_stats()