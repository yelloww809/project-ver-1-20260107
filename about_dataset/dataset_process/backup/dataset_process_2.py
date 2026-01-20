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
INPUT_DATASET_DIR = Path('/data/hwh_data_folder/dataset/train')  # 原始数据集路径
PROCESSED_ROOT_DIR = Path('/data/hwh_data_folder/processed_datasets') # 输出根目录
OUTPUT_FOLDER_NAME = 'yolo_train_set' # 预处理后数据集的文件夹名称

# --- 采样与划分设置 ---
TOTAL_SAMPLES = 100       # -1: 全部处理; >0: 随机选择多少个原始文件处理
RANDOM_SEED = 42
TRAIN_VAL_SPLIT_RATIO = 0.8  # 80% 原始文件用于训练，20% 用于验证

# --- 负样本掺入策略 ---
NEGATIVE_RATIO = 0.06     # 负切片保留比例 (相对于该文件产生的正切片数量)
                          # 例如: 某文件切出 100 个正切片，则随机保留 ceil(100*0.06) = 6 个负切片

# --- STFT 信号处理设置 ---
FREQ_RES_KHZ = 5           # 频率分辨率 (kHz)
OVERLAP_RATIO = 0.5        # 窗重叠率
USE_DB_SCALE = False       # True: dB; False: Linear (线性)

# --- 归一化设置 ---
NORM_TYPE = 'GLOBAL'       # 'GLOBAL'
GLOBAL_MIN_DB = -140.0     # 底噪 (dB)
GLOBAL_MAX_DB = 30.0       # 饱和点 (dB)

# --- 标签过滤设置 ---
TARGET_SIGNALS = {
    9:  [0.0523, 0.0625, 0.25],
    10: [0.3, 0.5],
    11: [1.6],
    12: [0.006, 0.2],
    13: [0.04, 0.12, 0.2]
}
BW_TOLERANCE = 0.002 

# --- 切片 (Slicing) 设置 ---
ENABLE_SLICING = True      
SLICE_HEIGHT = 640         
SLICE_WIDTH = 640          
SLICE_OVERLAP = 0.2        

# --- 可视化设置 ---
NUM_VISUAL_SAMPLES = 100   # 最后检查用的样本数

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
    """生成 ultralytics 训练所需的 yaml 文件"""
    yaml_path = output_dir / f"{OUTPUT_FOLDER_NAME}.yaml"
    
    # 自动获取所有可能的 class id (从0到13)
    # 也可以只写出现在 TARGET_SIGNALS 里的，但写全比较安全
    names_dict = {i: f"{i}" for i in range(14)}
    
    content = [
        f"path: {output_dir.absolute()}", # 使用绝对路径最稳妥
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
        f.write(f"Dataset Preprocessing Settings - {datetime.now()}\n")
        f.write("========================================\n")
        f.write(f"INPUT_DATASET_DIR: {INPUT_DATASET_DIR}\n")
        f.write(f"OUTPUT_FOLDER_NAME: {OUTPUT_FOLDER_NAME}\n")
        f.write(f"TOTAL_SAMPLES: {TOTAL_SAMPLES}\n")
        f.write(f"TRAIN_VAL_SPLIT_RATIO: {TRAIN_VAL_SPLIT_RATIO}\n")
        f.write(f"NEGATIVE_RATIO: {NEGATIVE_RATIO}\n")
        f.write(f"FREQ_RES_KHZ: {FREQ_RES_KHZ}\n")
        f.write(f"USE_DB_SCALE: {USE_DB_SCALE}\n")
        f.write(f"GLOBAL_MIN_DB: {GLOBAL_MIN_DB}\n")
        f.write(f"GLOBAL_MAX_DB: {GLOBAL_MAX_DB}\n")
        f.write(f"TARGET_SIGNALS: {json.dumps(TARGET_SIGNALS, indent=2)}\n")

# ==========================================
# 3. 主处理流程
# ==========================================

def main():
    # 1. 初始化
    output_dir = PROCESSED_ROOT_DIR / OUTPUT_FOLDER_NAME
    norm_min, norm_max = calculate_params(GLOBAL_MIN_DB, GLOBAL_MAX_DB, USE_DB_SCALE)
    
    print(f"--- 预处理开始 ---")
    print(f"模式: 按原始文件划分 Train/Valid")
    print(f"负样本掺入比例: {NEGATIVE_RATIO}")
    print(f"输出目录: {output_dir}")

    # 定义最终目录结构
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

    # 2. 文件采样与划分 (Split by Original File)
    all_bins = list(INPUT_DATASET_DIR.glob('*.bin'))
    if TOTAL_SAMPLES != -1 and TOTAL_SAMPLES < len(all_bins):
        random.seed(RANDOM_SEED)
        sample_bins = random.sample(all_bins, TOTAL_SAMPLES)
    else:
        sample_bins = all_bins
    
    # 随机打乱并按比例划分原始文件 ID
    random.seed(RANDOM_SEED)
    random.shuffle(sample_bins)
    
    split_idx = int(len(sample_bins) * TRAIN_VAL_SPLIT_RATIO)
    train_bins = sample_bins[:split_idx]
    valid_bins = sample_bins[split_idx:]
    
    print(f"总文件数: {len(sample_bins)}")
    print(f"训练集原始文件: {len(train_bins)} 个")
    print(f"验证集原始文件: {len(valid_bins)} 个")

    # 用于可视化的候选列表
    viz_candidates = [] 

    # 3. 核心处理函数 (处理一组文件并保存到指定 split 目录)
    def process_split_set(bin_files, img_dst_dir, lbl_dst_dir, desc_text):
        count_pos = 0
        count_neg = 0
        
        for bin_file in tqdm(bin_files, desc=desc_text):
            file_idx = int(bin_file.stem)
            json_path = INPUT_DATASET_DIR / f'{file_idx}.json'
            if not json_path.exists(): continue

            # --- A. 数据读取与 STFT ---
            with open(json_path, 'r') as f: meta = json.load(f)
            obs_range = meta['observation_range']
            f_min, f_max = obs_range[0], obs_range[1]
            bw_mhz = f_max - f_min
            fs = bw_mhz * 1e6
            
            raw_data = np.fromfile(bin_file, dtype=np.float16)
            signal = raw_data[::2] + 1j * raw_data[1::2]
            duration_sec = len(signal) / fs
            
            nperseg = int(fs / (FREQ_RES_KHZ * 1000))
            noverlap = int(nperseg * OVERLAP_RATIO)
            img_rgb, (full_h, full_w) = process_stft(signal, fs, nperseg, noverlap, USE_DB_SCALE, norm_min, norm_max)
            
            # --- B. 标签过滤 ---
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

            # --- C. 切片与分类 (内存操作，不存盘) ---
            file_pos_slices = [] # 存 (img, labels_str_list, name)
            file_neg_slices = [] # 存 (img, name)
            
            if ENABLE_SLICING:
                slices = get_slice_coordinates(full_h, full_w)
                for i, (sy1, sy2, sx1, sx2) in enumerate(slices):
                    img_slice = img_rgb[sy1:sy2, sx1:sx2]
                    slice_h, slice_w = sy2 - sy1, sx2 - sx1
                    
                    slice_labels = []
                    for box in valid_boxes:
                        bx1, by1, bx2, by2, cid = box
                        ix1, iy1, ix2, iy2 = max(bx1, sx1), max(by1, sy1), min(bx2, sx2), min(by2, sy2)
                        if ix2 > ix1 and iy2 > iy1:
                            cx, cy, w, h = convert_box_to_yolo((ix1-sx1, iy1-sy1, ix2-sx1, iy2-sy1), slice_w, slice_h)
                            slice_labels.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    
                    base_name = f"{file_idx}_s{i}"
                    if slice_labels:
                        file_pos_slices.append((img_slice, slice_labels, base_name))
                    else:
                        file_neg_slices.append((img_slice, base_name))
            else:
                # 全图模式 (逻辑同上，只是不切片)
                base_name = f"{file_idx}"
                yolo_lines = []
                for box in valid_boxes:
                    bx1, by1, bx2, by2, cid = box
                    cx, cy, w, h = convert_box_to_yolo((bx1, by1, bx2, by2), full_w, full_h)
                    yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                
                if yolo_lines:
                    file_pos_slices.append((img_rgb, yolo_lines, base_name))
                else:
                    file_neg_slices.append((img_rgb, base_name))

            # --- D. 实时采样与保存 ---
            
            # 1. 保存所有正切片
            for img, lbls, name in file_pos_slices:
                cv2.imwrite(str(img_dst_dir / f"{name}.png"), img)
                with open(lbl_dst_dir / f"{name}.txt", 'w') as f_lbl:
                    f_lbl.write('\n'.join(lbls))
                # 收集用于可视化
                viz_candidates.append((img_dst_dir / f"{name}.png", lbl_dst_dir / f"{name}.txt"))
                count_pos += 1
            
            # 2. 随机采样负切片
            # 目标数量 = 正切片数量 * 比例 (向上取整)
            # 如果该文件没有正切片，num_neg_to_keep 为 0 (或者你可以设置至少保留1个，这里暂按比例严格执行)
            num_pos_in_file = len(file_pos_slices)
            num_neg_to_keep = math.ceil(num_pos_in_file * NEGATIVE_RATIO)
            
            if num_neg_to_keep > 0 and len(file_neg_slices) > 0:
                selected_negs = random.sample(file_neg_slices, min(num_neg_to_keep, len(file_neg_slices)))
                
                for img, name in selected_negs:
                    # 保存图片
                    cv2.imwrite(str(img_dst_dir / f"{name}.png"), img)
                    # 保存空的标签文件 (Ultralytics 要求)
                    with open(lbl_dst_dir / f"{name}.txt", 'w') as f_lbl:
                        pass 
                    count_neg += 1

        print(f"[{desc_text}] 完成. 累计: 正样本 {count_pos}, 负样本 {count_neg}")

    # 4. 执行 Train 和 Valid 的处理
    process_split_set(train_bins, dirs['train_img'], dirs['train_lbl'], "Generating Train Set")
    process_split_set(valid_bins, dirs['valid_img'], dirs['valid_lbl'], "Generating Valid Set")

    # ==========================================
    # 5. 可视化检查
    # ==========================================
    print("\nStep 3: 生成可视化检查样本...")
    
    if len(viz_candidates) > 0:
        random.seed(RANDOM_SEED)
        viz_samples = random.sample(viz_candidates, min(NUM_VISUAL_SAMPLES, len(viz_candidates)))
        
        for img_p, lbl_p in viz_samples:
            img = cv2.imread(str(img_p))
            if img is None: continue
            h, w, _ = img.shape
            
            if lbl_p.exists():
                with open(lbl_p, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue # 跳过空行
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    box_w, box_h = bw * w, bh * h
                    x1 = int(cx * w - box_w / 2)
                    y1 = int(cy * h - box_h / 2)
                    x2 = int(cx * w + box_w / 2)
                    y2 = int(cy * h + box_h / 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"Class {cls_id}", (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imwrite(str(dirs['visual'] / f"vis_{img_p.name}"), img)
        print(f"已保存可视化图片至 {dirs['visual']}")
    
    print(f"\n全部流程结束！")

if __name__ == "__main__":
    main()