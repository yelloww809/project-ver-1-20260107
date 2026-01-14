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
from collections import Counter

# ==========================================
# 1. 人工设置参数 (USER CONFIGURATION)
# ==========================================

# --- 路径设置 ---
INPUT_DATASET_DIR = Path(r'E:\huangwenhao\dataset\train')
PROCESSED_ROOT_DIR = Path(r'E:\huangwenhao\processed_datasets\v8')

# [关键] 输出文件夹名称
OUTPUT_FOLDER_NAME = 'v8_large_jpg_2' 

# --- 图片保存格式设置 ---
SAVE_IMAGE_FORMAT = 'jpg' 

# --- 采样与数据集划分比例 ---
TOTAL_SAMPLES = -1        # -1: 全部处理
RANDOM_SEED = 42
# 训练集 : 验证集 : 测试集 (和必须为 1.0)
TRAIN_VAL_TEST_RATIO = (0.8, 0.1, 0.1)

# --- 负样本掺入策略 ---
NEGATIVE_RATIO = 0.06     

# --- STFT 信号处理设置 ---
USE_FIXED_RES_IN_FULL_MODE = False      # === Compare With v8_large_1.py ===      
FREQ_RES_KHZ = 20         
OVERLAP_RATIO = 0.5       
USE_DB_SCALE = True

# --- 归一化设置 ---
NORM_TYPE = 'GLOBAL'
GLOBAL_MIN_DB = -140.0
GLOBAL_MAX_DB = 30.0

# --- 标签过滤设置 ---
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
BW_TOLERANCE = 0.5 

# === small ===
# TARGET_SIGNALS = {
#   9:  [0.0523, 0.0625, 0.25],
#   10: [0.3, 0.5],
#   12: [0.006, 0.2],
#   13: [0.04, 0.12, 0.2]
# }
# BW_TOLERANCE = 0.002 

# --- 切片设置 ---
ENABLE_SLICING = False    

SLICE_HEIGHT = 640        
SLICE_WIDTH = 640          
SLICE_OVERLAP = 0.2        

# --- 可视化设置 ---
NUM_VISUAL_SAMPLES = 50

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
    f, t, Zxx = stft(iq_signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    magnitude = np.abs(Zxx)
    
    if use_db:
        data = 20 * np.log10(magnitude + 1e-12)
    else:
        data = magnitude

    if NORM_TYPE == 'GLOBAL':
        data = np.clip(data, norm_min, norm_max)
        data = (data - norm_min) / (norm_max - norm_min)
    else:
        local_min, local_max = data.min(), data.max()
        if local_max > local_min:
            data = (data - local_min) / (local_max - local_min)
        else:
            data = np.zeros_like(data)
            
    img_u8 = (data * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return img_rgb, img_u8.shape

def get_slice_coordinates(img_h, img_w):
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
    
    # 保持 0-13 范围 (V8 逻辑)
    names_dict = {i: f"{i}" for i in range(14)}
    
    content = [
        f"path: {output_dir.absolute()}", 
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        "",
        "names:",
    ]
    for k, v in names_dict.items():
        content.append(f"  {k}: {v}")
    
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(content))
    print(f"YAML 配置文件已生成: {yaml_path}")

def save_initial_settings(output_path):
    """保存初步设置"""
    with open(output_path, 'w') as f:
        f.write(f"Dataset Settings (v8 improved) - {datetime.now()}\n\n")
        f.write(f"total sample\t=\t{TOTAL_SAMPLES}\n")
        f.write(f"split ratio\t=\tTrain:{TRAIN_VAL_TEST_RATIO[0]} Val:{TRAIN_VAL_TEST_RATIO[1]} Test:{TRAIN_VAL_TEST_RATIO[2]}\n")
        f.write(f"format\t\t\t=\t{SAVE_IMAGE_FORMAT}\n")
        
        if USE_FIXED_RES_IN_FULL_MODE:
            f.write(f"freq resolution\t=\t{FREQ_RES_KHZ}\t(kHz)\n")
        else:
            f.write(f"freq resolution\t=\tAdaptive\n")
        
        if USE_DB_SCALE:
            f.write(f"dB or linear\t=\tdB\n")
        else:
            f.write(f"dB or linear\t=\tlinear\n")
            
        f.write(f"norm\t\t\t=\t{NORM_TYPE}\n")
        f.write(f"targets signals\t=\t{json.dumps(TARGET_SIGNALS)}\n")

# --- [新增] 分布检查与记录函数 ---
def perform_distribution_check(train_bins, valid_bins, test_bins, settings_path):
    print("\n>>> 开始执行分布预检查 (Scanning JSONs)...")
    
    def count_in_dataset(bin_list):
        c = Counter()
        # 仅用于统计，不进行耗时的STFT
        for bin_file in bin_list:
            json_path = INPUT_DATASET_DIR / f'{bin_file.stem}.json'
            if not json_path.exists(): continue
            try:
                with open(json_path, 'r') as f: meta = json.load(f)
                for sig in meta.get('signals', []):
                    # 必须应用相同的过滤逻辑
                    obs_range = meta['observation_range']
                    # 注意：如果json中频率单位不是MHz，需根据实际情况调整。假设逻辑与 process_split_set 一致。
                    bw = sig['end_frequency'] - sig['start_frequency']
                    if is_target_signal(sig['class'], bw):
                        c[sig['class']] += 1
            except Exception:
                pass
        return c

    # 统计
    train_counts = count_in_dataset(train_bins)
    valid_counts = count_in_dataset(valid_bins)
    test_counts = count_in_dataset(test_bins)
    
    # 所有的目标ID
    target_ids = sorted(list(TARGET_SIGNALS.keys()))
    
    # 1. 打印到控制台
    header = f"{'Class ID':<10} | {'Train':<10} | {'Val':<10} | {'Test':<10}"
    print("-" * 50)
    print(header)
    print("-" * 50)
    
    for cid in target_ids:
        print(f"{cid:<10} | {train_counts[cid]:<10} | {valid_counts[cid]:<10} | {test_counts[cid]:<10}")
    print("-" * 50)
    print(">>> 预检查完成。\n")

    # 2. 追加到 settings.txt
    with open(settings_path, 'a') as f:
        f.write("\n\n" + "="*40 + "\n")
        f.write("Class Distribution Statistics\n")
        f.write("="*40 + "\n")
        f.write(f"{'Class ID':<10}\t{'Train':<10}\t{'Val':<10}\t{'Test':<10}\n")
        for cid in target_ids:
            f.write(f"{cid:<10}\t{train_counts[cid]:<10}\t{valid_counts[cid]:<10}\t{test_counts[cid]:<10}\n")
        f.write("="*40 + "\n")
    
    print(f"统计信息已追加至: {settings_path}")

# ==========================================
# 3. 主处理流程
# ==========================================

def main():
    output_dir = PROCESSED_ROOT_DIR / OUTPUT_FOLDER_NAME
    norm_min, norm_max = calculate_params(GLOBAL_MIN_DB, GLOBAL_MAX_DB, USE_DB_SCALE)
    
    img_ext = '.jpg' if SAVE_IMAGE_FORMAT.lower() == 'jpg' else '.png'
    
    print(f"--- 预处理开始 (v8 Improved) ---")
    print(f"图片保存格式: {SAVE_IMAGE_FORMAT.upper()} ({img_ext})")
    
    if abs(sum(TRAIN_VAL_TEST_RATIO) - 1.0) > 1e-6:
        print("Error: 训练/验证/测试比例之和不为 1！")
        return

    dirs = {
        'train_img': output_dir / 'images/train',
        'valid_img': output_dir / 'images/valid',
        'test_img':  output_dir / 'images/test',
        'train_lbl': output_dir / 'labels/train',
        'valid_lbl': output_dir / 'labels/valid',
        'test_lbl':  output_dir / 'labels/test',
        'visual': output_dir / 'visual'
    }
    ensure_dirs(dirs.values())
    
    # 保存初始设置
    settings_path = output_dir / 'settings.txt'
    save_initial_settings(settings_path)
    
    generate_yaml(output_dir)

    all_bins = list(INPUT_DATASET_DIR.glob('*.bin'))
    if TOTAL_SAMPLES != -1 and TOTAL_SAMPLES < len(all_bins):
        random.seed(RANDOM_SEED)
        sample_bins = random.sample(all_bins, TOTAL_SAMPLES)
    else:
        sample_bins = all_bins
    
    # 保持 V8 的简单随机 shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(sample_bins)
    
    n_total = len(sample_bins)
    n_train = int(n_total * TRAIN_VAL_TEST_RATIO[0])
    n_val = int(n_total * TRAIN_VAL_TEST_RATIO[1])
    
    train_bins = sample_bins[:n_train]
    valid_bins = sample_bins[n_train : n_train + n_val]
    test_bins = sample_bins[n_train + n_val :]
    
    print(f"划分结果: Train={len(train_bins)}, Val={len(valid_bins)}, Test={len(test_bins)}")

    # [新增] 执行分布检查并写入文件
    perform_distribution_check(train_bins, valid_bins, test_bins, settings_path)

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
            
            raw_data = np.fromfile(bin_file, dtype=np.float16)
            signal = raw_data[::2] + 1j * raw_data[1::2]
            signal_len = len(signal)
            duration_sec = signal_len / fs
            
            use_fixed_for_this_run = False
            if ENABLE_SLICING:
                use_fixed_for_this_run = True
            elif USE_FIXED_RES_IN_FULL_MODE:
                use_fixed_for_this_run = True
            
            if use_fixed_for_this_run:
                nperseg = int(fs / (FREQ_RES_KHZ * 1000))
                if nperseg > signal_len: nperseg = signal_len
            else:
                try:
                    calculated_nperseg = int(math.sqrt(signal_len / (1 - OVERLAP_RATIO)))
                    if calculated_nperseg % 2 != 0: calculated_nperseg += 1
                    nperseg = min(calculated_nperseg, signal_len)
                    nperseg = max(nperseg, 64) 
                except ValueError:
                    nperseg = 256

            noverlap = int(nperseg * OVERLAP_RATIO)
            img_rgb, (full_h, full_w) = process_stft(signal, fs, nperseg, noverlap, USE_DB_SCALE, norm_min, norm_max)
            
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
                    img_slice = img_rgb[sy1:sy2, sx1:sx2]
                    save_path = img_dst_dir / f"{base_name}{img_ext}"
                    lbl_path = lbl_dst_dir / f"{base_name}.txt"
                    
                    cv2.imwrite(str(save_path), img_slice)
                    with open(lbl_path, 'w') as f_lbl:
                        f_lbl.write('\n'.join(slice_labels))
                    
                    viz_candidates.append((save_path, lbl_path))
                    file_pos_count += 1
                    count_pos += 1
                else:
                    img_slice = img_rgb[sy1:sy2, sx1:sx2]
                    file_neg_buffer.append((img_slice, base_name))
            
            neg_quota_balance += file_pos_count * NEGATIVE_RATIO
            num_to_save = int(neg_quota_balance)
            
            if num_to_save > 0 and file_neg_buffer:
                actual_save = min(num_to_save, len(file_neg_buffer))
                selected = random.sample(file_neg_buffer, actual_save)
                for img, name in selected:
                    save_path = img_dst_dir / f"{name}{img_ext}"
                    cv2.imwrite(str(save_path), img)
                    with open(lbl_dst_dir / f"{name}.txt", 'w') as f_lbl: pass 
                    count_neg += 1
                neg_quota_balance -= actual_save
            
            del file_neg_buffer, img_rgb, raw_data

        print(f"[{desc_text}] 完成. 累计: 正样本 {count_pos}, 负样本 {count_neg}")

    process_split_set(train_bins, dirs['train_img'], dirs['train_lbl'], "Generating Train Set")
    process_split_set(valid_bins, dirs['valid_img'], dirs['valid_lbl'], "Generating Valid Set")
    process_split_set(test_bins,  dirs['test_img'],  dirs['test_lbl'],  "Generating Test Set")

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
            cv2.imwrite(str(dirs['visual'] / f"vis_{img_p.stem}.jpg"), img)
    
    print(f"\n全部流程结束！")

if __name__ == "__main__":
    main()