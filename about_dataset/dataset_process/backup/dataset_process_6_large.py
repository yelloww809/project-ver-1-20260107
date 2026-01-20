import numpy as np
import json
import os
import shutil
import cv2
import random
import math
from scipy.signal import stft
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ==========================================
# 1. 人工设置参数 (USER CONFIGURATION)
# ==========================================

# --- 路径设置 ---
INPUT_DATASET_DIR = Path(r'E:\huangwenhao\dataset\train')
PROCESSED_ROOT_DIR = Path(r'E:\huangwenhao\processed_datasets')

# [关键] 输出文件夹名称
OUTPUT_FOLDER_NAME = 'yolo_train_set_v6_3000_large' 

# --- 采样与划分设置 ---
TOTAL_SAMPLES = 3000        # -1: 全部处理
RANDOM_SEED = 42
TRAIN_VAL_SPLIT_RATIO = 0.8 

# --- 负样本掺入策略 ---
NEGATIVE_RATIO = 0.06     # 累积配额制，正样本的 6%

# --- STFT 信号处理设置 ---
FREQ_RES_KHZ = 20          # 频率分辨率 (kHz)
OVERLAP_RATIO = 0.5       
USE_DB_SCALE = True

# --- 归一化设置 ---
NORM_TYPE = 'GLOBAL'
GLOBAL_MIN_DB = -140.0
GLOBAL_MAX_DB = 30.0

# --- [关键] 标签过滤设置 ---
# === big ===
TARGET_SIGNALS = {
    0:  [20.0],
    1:  [20.0],
    2:  [20.0],
    3:  [20.0, 40.0],
    4:  [40.0],
    5:  [40.0],
    6:  [1.0],
    7:  [2.0],
    8:  [2.0],
    10: [10.0],
    11: [1.6, 7.56, 10.0],
}
BW_TOLERANCE = 0.5 # [注意] 宽带信号的带宽误差可能会大一点，这里设宽一点 (0.5MHz) 比较安全

# # === small ===
# TARGET_SIGNALS = {
#     9:  [0.0523, 0.0625, 0.25],
#     10: [0.3, 0.5],
#     12: [0.006, 0.2],
#     13: [0.04, 0.12, 0.2]
# }
# BW_TOLERANCE = 0.002 

# --- [关键] 切片 (Slicing) 设置 ---
ENABLE_SLICING = False     # False = 全图; True = 切片

# --- [关键] 全图模式下的 STFT 策略 (v6 新增) ---
# 仅当 ENABLE_SLICING = False 时生效
# True:  强制使用 FREQ_RES_KHZ (保证信号内部纹理一致，但图片可能长宽比极端) -> 推荐
# False: 自适应计算窗长 (保证图片接近正方形，但信号内部纹理会变)
USE_FIXED_RES_IN_FULL_MODE = False 

SLICE_HEIGHT = 640         
SLICE_WIDTH = 640          
SLICE_OVERLAP = 0.2        

# --- 可视化设置 ---
NUM_VISUAL_SAMPLES = 100

# ==========================================
# 2. 辅助函数定义
# ==========================================

def ensure_dirs(path_list):
    for p in path_list:
        p.mkdir(parents=True, exist_ok=True)

def calculate_params(min_db, max_db, use_db):
    if use_db:
        return min_db, max_db
    else:
        min_linear = 10 ** (min_db / 20.0)
        max_linear = 10 ** (max_db / 20.0)
        return min_linear, max_linear

def is_target_signal(cls_id, bandwidth):
    if cls_id not in TARGET_SIGNALS:
        return False
    allowed_bws = TARGET_SIGNALS[cls_id]
    for bw in allowed_bws:
        if abs(bandwidth - bw) <= BW_TOLERANCE:
            return True
    return False

def process_stft(iq_signal, fs, nperseg, noverlap, use_db, norm_min, norm_max):
    # 执行 STFT
    f, t, Zxx = stft(iq_signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    magnitude = np.abs(Zxx)
    
    # 转 dB 或 保持线性
    if use_db:
        data = 20 * np.log10(magnitude + 1e-12)
    else:
        data = magnitude

    # 归一化
    if NORM_TYPE == 'GLOBAL':
        data = np.clip(data, norm_min, norm_max)
        data = (data - norm_min) / (norm_max - norm_min)
    else:
        local_min, local_max = data.min(), data.max()
        if local_max > local_min:
            data = (data - local_min) / (local_max - local_min)
        else:
            data = np.zeros_like(data)
            
    # 转图片
    img_u8 = (data * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return img_rgb, img_u8.shape

def get_slice_coordinates(img_h, img_w):
    # 如果不切片，返回唯一的全图坐标
    if not ENABLE_SLICING:
        return [(0, img_h, 0, img_w)]

    stride_h = int(SLICE_HEIGHT * (1 - SLICE_OVERLAP))
    stride_w = int(SLICE_WIDTH * (1 - SLICE_OVERLAP))
    y_starts = list(range(0, img_h, stride_h))
    x_starts = list(range(0, img_w, stride_w))
    slices = []
    
    for y in y_starts:
        y_end = min(y + SLICE_HEIGHT, img_h)
        if y >= img_h: continue
        for x in x_starts:
            x_end = min(x + SLICE_WIDTH, img_w)
            if x >= img_w: continue
            slices.append((y, y_end, x, x_end))
            if x_end == img_w: break
        if y_end == img_h: break
    return slices

def convert_box_to_yolo(box_px, img_w, img_h):
    x1, y1, x2, y2 = box_px
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx * dw, cy * dh, w * dw, h * dh

def generate_yaml(output_dir):
    yaml_path = output_dir / f"{OUTPUT_FOLDER_NAME}.yaml"
    names_dict = {i: f"{i}" for i in range(14)}
    content = [
        f"path: {output_dir.absolute()}", 
        "train: images/train",
        "val: images/valid",
        "",
        "names:",
    ]
    for k, v in names_dict.items():
        content.append(f"  {k}: {v}")
    
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(content))
    print(f"YAML 配置文件已生成: {yaml_path}")

def save_settings(output_path):
    with open(output_path, 'w') as f:
        f.write(f"Dataset Settings (v6) - {datetime.now()}\n")
        f.write(f"ENABLE_SLICING: {ENABLE_SLICING}\n")
        f.write(f"USE_FIXED_RES_IN_FULL_MODE: {USE_FIXED_RES_IN_FULL_MODE}\n")
        f.write(f"TARGET_SIGNALS: {json.dumps(TARGET_SIGNALS)}\n")
        f.write(f"TOTAL_SAMPLES: {TOTAL_SAMPLES}\n")
        f.write(f"NEGATIVE_RATIO: {NEGATIVE_RATIO}\n")
        f.write(f"NORM_TYPE: {NORM_TYPE}\n")
        f.write(f"FREQ_RES_KHZ: {FREQ_RES_KHZ} (if Used)\n")

# ==========================================
# 3. 主处理流程
# ==========================================

def main():
    output_dir = PROCESSED_ROOT_DIR / OUTPUT_FOLDER_NAME
    norm_min, norm_max = calculate_params(GLOBAL_MIN_DB, GLOBAL_MAX_DB, USE_DB_SCALE)
    
    print(f"--- 预处理开始 (v6) ---")
    
    # 逻辑判断打印
    use_fixed_res = False
    if ENABLE_SLICING:
        print(f"模式: 切片处理 | 分辨率: 固定 {FREQ_RES_KHZ} kHz")
        use_fixed_res = True
    else:
        if USE_FIXED_RES_IN_FULL_MODE:
            print(f"模式: 全图处理 | 分辨率: 固定 {FREQ_RES_KHZ} kHz (特征一致性优先)")
            use_fixed_res = True
        else:
            print(f"模式: 全图处理 | 分辨率: 自适应方形窗 (几何比例优先)")
            use_fixed_res = False

    print(f"输出目录: {output_dir}")
    print(f"关注信号: {list(TARGET_SIGNALS.keys())}")

    dirs = {
        'train_img': output_dir / 'images/train',
        'valid_img': output_dir / 'images/valid',
        'train_lbl': output_dir / 'labels/train',
        'valid_lbl': output_dir / 'labels/valid',
        'visual': output_dir / 'visual'
    }
    ensure_dirs(dirs.values())
    save_settings(output_dir / 'settings.txt')
    generate_yaml(output_dir)

    all_bins = list(INPUT_DATASET_DIR.glob('*.bin'))
    if TOTAL_SAMPLES != -1 and TOTAL_SAMPLES < len(all_bins):
        random.seed(RANDOM_SEED)
        sample_bins = random.sample(all_bins, TOTAL_SAMPLES)
    else:
        sample_bins = all_bins
    
    random.seed(RANDOM_SEED)
    random.shuffle(sample_bins)
    split_idx = int(len(sample_bins) * TRAIN_VAL_SPLIT_RATIO)
    train_bins = sample_bins[:split_idx]
    valid_bins = sample_bins[split_idx:]
    
    viz_candidates = [] 

    def process_split_set(bin_files, img_dst_dir, lbl_dst_dir, desc_text):
        count_pos = 0
        count_neg = 0
        neg_quota_balance = 0.0 
        
        for bin_file in tqdm(bin_files, desc=desc_text):
            file_idx = int(bin_file.stem)
            json_path = INPUT_DATASET_DIR / f'{file_idx}.json'
            if not json_path.exists(): continue

            with open(json_path, 'r') as f: meta = json.load(f)
            obs_range = meta['observation_range']
            f_min, f_max = obs_range[0], obs_range[1]
            bw_mhz = f_max - f_min
            fs = bw_mhz * 1e6
            
            # --- 1. 读取信号 ---
            raw_data = np.fromfile(bin_file, dtype=np.float16)
            signal = raw_data[::2] + 1j * raw_data[1::2]
            signal_len = len(signal)
            duration_sec = signal_len / fs
            
            # --- 2. 确定 STFT 参数 (v6 逻辑) ---
            # 判断是否使用固定分辨率
            use_fixed_for_this_run = False
            if ENABLE_SLICING:
                use_fixed_for_this_run = True
            elif USE_FIXED_RES_IN_FULL_MODE:
                use_fixed_for_this_run = True
            
            if use_fixed_for_this_run:
                # 模式 A: 固定分辨率 (保证纹理一致)
                nperseg = int(fs / (FREQ_RES_KHZ * 1000))
                # 保护措施：防止 nperseg 大于信号长度
                if nperseg > signal_len:
                    nperseg = signal_len
            else:
                # 模式 B: 自适应方形窗 (保证图片形状)
                try:
                    calculated_nperseg = int(math.sqrt(signal_len / (1 - OVERLAP_RATIO)))
                    if calculated_nperseg % 2 != 0: calculated_nperseg += 1
                    nperseg = min(calculated_nperseg, signal_len)
                    nperseg = max(nperseg, 64) 
                except ValueError:
                    nperseg = 256

            noverlap = int(nperseg * OVERLAP_RATIO)
            
            # --- 3. 执行 STFT ---
            img_rgb, (full_h, full_w) = process_stft(signal, fs, nperseg, noverlap, USE_DB_SCALE, norm_min, norm_max)
            
            # --- 4. 标签解析 ---
            valid_boxes = [] 
            for sig in meta.get('signals', []):
                if is_target_signal(sig['class'], sig['end_frequency'] - sig['start_frequency']):
                    x1 = (sig['start_time'] / 1000.0 / duration_sec) * full_w
                    x2 = (sig['end_time'] / 1000.0 / duration_sec) * full_w
                    y1 = ((sig['start_frequency'] - f_min) / bw_mhz) * full_h
                    y2 = ((sig['end_frequency'] - f_min) / bw_mhz) * full_h
                    x1, x2 = np.clip([x1, x2], 0, full_w)
                    y1, y2 = np.clip([y1, y2], 0, full_h)
                    if x2 > x1 and y2 > y1:
                        valid_boxes.append([x1, y1, x2, y2, sig['class']])

            # --- 5. 切片与分类 ---
            file_pos_count = 0
            file_neg_buffer = [] 
            
            slice_coords = get_slice_coordinates(full_h, full_w)

            for i, (sy1, sy2, sx1, sx2) in enumerate(slice_coords):
                slice_h, slice_w = sy2 - sy1, sx2 - sx1
                
                slice_labels = []
                for box in valid_boxes:
                    bx1, by1, bx2, by2, cid = box
                    ix1, iy1, ix2, iy2 = max(bx1, sx1), max(by1, sy1), min(bx2, sx2), min(by2, sy2)
                    if ix2 > ix1 and iy2 > iy1:
                        cx, cy, w, h = convert_box_to_yolo((ix1-sx1, iy1-sy1, ix2-sx1, iy2-sy1), slice_w, slice_h)
                        slice_labels.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                
                base_name = f"{file_idx}" if not ENABLE_SLICING else f"{file_idx}_s{i}"
                
                if slice_labels:
                    # [正样本] -> 保存
                    img_slice = img_rgb[sy1:sy2, sx1:sx2]
                    cv2.imwrite(str(img_dst_dir / f"{base_name}.png"), img_slice)
                    with open(lbl_dst_dir / f"{base_name}.txt", 'w') as f_lbl:
                        f_lbl.write('\n'.join(slice_labels))
                    viz_candidates.append((img_dst_dir / f"{base_name}.png", lbl_dst_dir / f"{base_name}.txt"))
                    file_pos_count += 1
                    count_pos += 1
                else:
                    # [负样本] -> 缓冲
                    img_slice = img_rgb[sy1:sy2, sx1:sx2]
                    file_neg_buffer.append((img_slice, base_name))
            
            # --- 6. 负样本配额结算 ---
            neg_quota_balance += file_pos_count * NEGATIVE_RATIO
            num_to_save = int(neg_quota_balance)
            
            if num_to_save > 0 and file_neg_buffer:
                actual_save = min(num_to_save, len(file_neg_buffer))
                selected = random.sample(file_neg_buffer, actual_save)
                for img, name in selected:
                    cv2.imwrite(str(img_dst_dir / f"{name}.png"), img)
                    with open(lbl_dst_dir / f"{name}.txt", 'w') as f_lbl: pass 
                    count_neg += 1
                neg_quota_balance -= actual_save
            
            del file_neg_buffer, img_rgb, raw_data

        print(f"[{desc_text}] 完成. 累计: 正样本 {count_pos}, 负样本 {count_neg}")

    process_split_set(train_bins, dirs['train_img'], dirs['train_lbl'], "Generating Train Set")
    process_split_set(valid_bins, dirs['valid_img'], dirs['valid_lbl'], "Generating Valid Set")

    # 可视化
    print("\nStep 3: Generating Visualization...")
    if viz_candidates:
        random.seed(RANDOM_SEED)
        viz_samples = random.sample(viz_candidates, min(NUM_VISUAL_SAMPLES, len(viz_candidates)))
        for img_p, lbl_p in viz_samples:
            img = cv2.imread(str(img_p))
            if img is None: continue
            h, w, _ = img.shape
            if lbl_p.exists():
                with open(lbl_p, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, f"{cls_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(dirs['visual'] / f"vis_{img_p.name}"), img)
    
    print(f"\n全部流程结束！")

if __name__ == "__main__":
    main()