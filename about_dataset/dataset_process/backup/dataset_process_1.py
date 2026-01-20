import numpy as np
import json
import os
import shutil
import cv2
import random
from scipy.signal import stft
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. 人工设置参数 (USER CONFIGURATION)
# ==========================================

# --- 路径设置 ---
INPUT_DATASET_DIR = Path('/data/hwh_data_folder/dataset/train')  # 原始数据集路径
PROCESSED_ROOT_DIR = Path('/data/hwh_data_folder/processed_datasets') # 输出根目录
OUTPUT_FOLDER_NAME = 'yolo_train_set' # 预处理后数据集的文件夹名称

# --- 采样设置 ---
# 取值 -1 代表全部处理；取值 > 0 (如 100) 代表随机选择 100 个样本处理
TOTAL_SAMPLES = 100
RANDOM_SEED = 42

# --- STFT 信号处理设置 ---
FREQ_RES_KHZ = 5           # 频率分辨率 (kHz) => 决定 nperseg
OVERLAP_RATIO = 0.5        # 窗重叠率
USE_DB_SCALE = True       # True: 使用 dB 值; False: 使用线性值 (Linear Magnitude)

# --- 归一化设置 ---
NORM_TYPE = 'GLOBAL'       # 'GLOBAL' (全局归一化) 或 'SAMPLE' (样本内归一化)

# 全局归一化参数 (单位 dB)
#即使上方 USE_DB_SCALE = False，这里也请填写 dB 值，代码会自动转换
GLOBAL_MIN_DB = -140.0     # 对应线性值的底噪
GLOBAL_MAX_DB = 30.0       # 对应线性值的饱和点

# --- 标签过滤设置 ---
# 格式: {Class_ID: [带宽1(MHz), 带宽2(MHz), ...]}
# 只有匹配 Class ID 且 带宽在列表中的信号才会被记录，其余视为背景
TARGET_SIGNALS = {
    9:  [0.0523, 0.0625, 0.25],
    10: [0.3, 0.5],
    11: [1.6],
    12: [0.006, 0.2],
    13: [0.04, 0.12, 0.2]
}
BW_TOLERANCE = 0.002 # 带宽匹配容差 (MHz)，防止浮点数误差

# --- 切片 (Slicing) 设置 ---
ENABLE_SLICING = True      # 是否进行切片
SLICE_HEIGHT = 640         # 切片高度 (对应频率轴)
SLICE_WIDTH = 640          # 切片宽度 (对应时间轴)
SLICE_OVERLAP = 0.2        # 切片重叠度 (20%)

# --- 可视化设置 ---
NUM_VISUAL_SAMPLES = 100   # 可视化检查的样本数量

# ==========================================
# 2. 辅助函数定义
# ==========================================

def ensure_dirs(path_list):
    """创建所需的文件夹"""
    for p in path_list:
        p.mkdir(parents=True, exist_ok=True)

def calculate_params(min_db, max_db, use_db):
    """计算归一化的数值边界"""
    if use_db:
        return min_db, max_db
    else:
        # 线性模式：将 dB 转为线性幅度 (Amplitude)
        # dB = 20 * log10(Amp)  =>  Amp = 10^(dB/20)
        min_linear = 10 ** (min_db / 20.0)
        max_linear = 10 ** (max_db / 20.0)
        return min_linear, max_linear

def is_target_signal(cls_id, bandwidth):
    """判断信号是否在关注列表中"""
    if cls_id not in TARGET_SIGNALS:
        return False
    
    allowed_bws = TARGET_SIGNALS[cls_id]
    for bw in allowed_bws:
        if abs(bandwidth - bw) <= BW_TOLERANCE:
            return True
    return False

def process_stft(iq_signal, fs, nperseg, noverlap, use_db, norm_min, norm_max):
    """执行 STFT 并归一化转为图像"""
    # 1. STFT
    # return_onesided=False 得到双边谱，我们需要 shift 将 0Hz 移到中心还是?
    # 原始需求：频率轴从上到下递增。
    # scipy stft 默认输出: frequencies 0, ..., fs/2. 
    # 如果不做 fftshift，0Hz (低频) 在 index 0 (Top)。这是符合需求的。
    # 只有当信号是复数基带信号且我们需要看负频率时才需要 shift。
    # 这里我们按照"低频在上"的标准处理。
    
    f, t, Zxx = stft(iq_signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    
    # 针对复数信号，fftshift 是必须的，将 0Hz 移到中心
    # 移位后：索引 0 是 -fs/2, 中间是 0, 最后是 +fs/2
    # 这样看起来：上部是负频率，中部是直流，下部是正频率。
    # 但根据题目 "频率轴从上到下递增"，即 顶部=MinFreq, 底部=MaxFreq。
    # fftshift 后的数组正好满足：index 0 (Top) 是最小频率，index -1 (Bottom) 是最大频率。
    Zxx = np.fft.fftshift(Zxx, axes=0)
    
    # 2. 取模
    magnitude = np.abs(Zxx)
    
    # 3. 数值转换 (Linear or dB)
    if use_db:
        data = 20 * np.log10(magnitude + 1e-12)
    else:
        data = magnitude

    # 4. 归一化
    if NORM_TYPE == 'GLOBAL':
        data = np.clip(data, norm_min, norm_max)
        data = (data - norm_min) / (norm_max - norm_min)
    else: # SAMPLE
        local_min, local_max = data.min(), data.max()
        if local_max > local_min:
            data = (data - local_min) / (local_max - local_min)
        else:
            data = np.zeros_like(data)
            
    # 5. 转为 8-bit 图像 (H, W)
    img_u8 = (data * 255).astype(np.uint8)
    
    # 6. 转为 3 通道 (RGB 兼容 YOLO)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    
    return img_rgb, img_u8.shape

def get_slice_coordinates(img_h, img_w):
    """
    生成切片坐标 (y1, y2, x1, x2)
    策略：硬性步长。如果最后一块小于 slice_size，则保留实际大小，不强制填充。
    """
    stride_h = int(SLICE_HEIGHT * (1 - SLICE_OVERLAP))
    stride_w = int(SLICE_WIDTH * (1 - SLICE_OVERLAP))
    
    y_starts = list(range(0, img_h, stride_h))
    x_starts = list(range(0, img_w, stride_w))
    
    slices = []
    
    # Y轴处理 (频率轴)
    for y in y_starts:
        y_end = min(y + SLICE_HEIGHT, img_h)
        # 如果这一块已经被前面的切片完全包含(不太可能发生)或者是空的，跳过
        if y >= img_h: continue
        
        # X轴处理 (时间轴)
        for x in x_starts:
            x_end = min(x + SLICE_WIDTH, img_w)
            if x >= img_w: continue
            
            slices.append((y, y_end, x, x_end))
            
            # 如果刚切的一块已经是最后一块了，停止内循环
            if x_end == img_w:
                break
        
        # 如果刚切的一行已经是最后一行了，停止外循环
        if y_end == img_h:
            break
            
    return slices

def convert_box_to_yolo(box_px, img_w, img_h):
    """像素坐标 (x1, y1, x2, y2) 转 YOLO (cx, cy, w, h)"""
    x1, y1, x2, y2 = box_px
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    
    return cx * dw, cy * dh, w * dw, h * dh

# ==========================================
# 3. 主处理流程
# ==========================================

def main():
    # 1. 准备参数
    output_dir = PROCESSED_ROOT_DIR / OUTPUT_FOLDER_NAME
    norm_min, norm_max = calculate_params(GLOBAL_MIN_DB, GLOBAL_MAX_DB, USE_DB_SCALE)
    
    print(f"--- 预处理开始 ---")
    print(f"模式: {'切片' if ENABLE_SLICING else '全图'}")
    print(f"数值: {'dB' if USE_DB_SCALE else 'Linear'}")
    print(f"归一化范围: {norm_min:.2e} ~ {norm_max:.2e}")
    print(f"输出目录: {output_dir}")
    
    # 2. 创建目录结构
    if output_dir.exists():
        print("警告: 输出目录已存在，可能会覆盖文件。")
    
    img_dir = output_dir / 'images'
    lbl_dir = output_dir / 'labels'
    vis_dir = output_dir / 'visual'

    pos_img_dir = img_dir / 'positive'
    neg_img_dir = img_dir / 'negative'
    pos_lbl_dir = lbl_dir / 'positive' # 负样本不需要标签文件
    
    if ENABLE_SLICING:
        ensure_dirs([pos_img_dir, neg_img_dir, pos_lbl_dir, vis_dir])
    else:
        ensure_dirs([img_dir, lbl_dir, vis_dir])

    # 3. 获取文件列表
    all_bins = list(INPUT_DATASET_DIR.glob('*.bin'))
    if TOTAL_SAMPLES != -1 and TOTAL_SAMPLES < len(all_bins):
        random.seed(RANDOM_SEED)
        sample_bins = random.sample(all_bins, TOTAL_SAMPLES)
    else:
        sample_bins = all_bins
    
    sample_indices = sorted([int(p.stem) for p in sample_bins])
    print(f"将处理 {len(sample_indices)} 个文件。")

    # 用于可视化的列表
    viz_candidates = [] # 存 tuple: (img_path, label_path)

    # 4. 循环处理
    for file_idx in tqdm(sample_indices, desc="Processing"):
        bin_path = INPUT_DATASET_DIR / f'{file_idx}.bin'
        json_path = INPUT_DATASET_DIR / f'{file_idx}.json'
        
        if not bin_path.exists() or not json_path.exists():
            continue
            
        # --- A. 读取数据 ---
        with open(json_path, 'r') as f:
            meta = json.load(f)
            
        obs_range = meta['observation_range']
        f_min, f_max = obs_range[0], obs_range[1]
        bw_mhz = f_max - f_min
        fs = bw_mhz * 1e6
        
        raw_data = np.fromfile(bin_path, dtype=np.float16)
        signal = raw_data[::2] + 1j * raw_data[1::2]
        duration_sec = len(signal) / fs
        
        # --- B. STFT 处理 ---
        nperseg = int(fs / (FREQ_RES_KHZ * 1000))
        noverlap = int(nperseg * OVERLAP_RATIO)
        
        # img_rgb: (H, W, 3) 
        # H 是频率轴 (Top=Min, Bottom=Max), W 是时间轴
        img_rgb, (full_h, full_w) = process_stft(
            signal, fs, nperseg, noverlap, USE_DB_SCALE, norm_min, norm_max
        )
        
        # --- C. 解析并过滤标签 ---
        valid_boxes = [] # 存像素坐标 [x1, y1, x2, y2, class_id]
        
        for sig in meta.get('signals', []):
            cls_id = sig['class']
            start_f = sig['start_frequency']
            end_f = sig['end_frequency']
            sig_bw = end_f - start_f
            
            # 过滤逻辑
            if is_target_signal(cls_id, sig_bw):
                start_t = sig['start_time'] / 1000.0 # ms -> s
                end_t = sig['end_time'] / 1000.0
                
                # 映射到图像坐标
                # 时间轴 (X)
                x1 = (start_t / duration_sec) * full_w
                x2 = (end_t / duration_sec) * full_w
                
                # 频率轴 (Y)
                # 因为 STFT fftshift 后，Top(0) 是 MinFreq, Bottom(H) 是 MaxFreq
                # 所以坐标是从上到下递增的，直接线性映射即可
                y1 = ((start_f - f_min) / bw_mhz) * full_h
                y2 = ((end_f - f_min) / bw_mhz) * full_h
                
                # 边界保护
                x1, x2 = np.clip([x1, x2], 0, full_w)
                y1, y2 = np.clip([y1, y2], 0, full_h)
                
                if x2 > x1 and y2 > y1:
                    valid_boxes.append([x1, y1, x2, y2, cls_id])
        
        # --- D. 保存 (切片 vs 不切片) ---
        
        if not ENABLE_SLICING:
            # === 不切片模式 ===
            save_name = f"{file_idx}"
            img_out_path = img_dir / f"{save_name}.png"
            lbl_out_path = lbl_dir / f"{save_name}.txt"
            
            # 保存图片
            cv2.imwrite(str(img_out_path), img_rgb)
            
            # 保存标签
            yolo_lines = []
            for box in valid_boxes:
                bx1, by1, bx2, by2, cid = box
                cx, cy, w, h = convert_box_to_yolo((bx1, by1, bx2, by2), full_w, full_h)
                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            with open(lbl_out_path, 'w') as f_lbl:
                f_lbl.write('\n'.join(yolo_lines))
                
            viz_candidates.append((img_out_path, lbl_out_path))
            
        else:
            # === 切片模式 ===
            slices = get_slice_coordinates(full_h, full_w)
            
            for i, (sy1, sy2, sx1, sx2) in enumerate(slices):
                slice_h = sy2 - sy1
                slice_w = sx2 - sx1
                
                # 提取切片图像
                img_slice = img_rgb[sy1:sy2, sx1:sx2]
                
                # 处理切片内的标签
                slice_labels = []
                for box in valid_boxes:
                    bx1, by1, bx2, by2, cid = box
                    
                    # 计算交集
                    ix1 = max(bx1, sx1)
                    iy1 = max(by1, sy1)
                    ix2 = min(bx2, sx2)
                    iy2 = min(by2, sy2)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        # 转换到切片局部坐标
                        local_x1 = ix1 - sx1
                        local_y1 = iy1 - sy1
                        local_x2 = ix2 - sx1
                        local_y2 = iy2 - sy1
                        
                        # 转 YOLO (相对于切片尺寸)
                        cx, cy, w, h = convert_box_to_yolo((local_x1, local_y1, local_x2, local_y2), slice_w, slice_h)
                        slice_labels.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                
                # 判定正负样本
                is_positive = len(slice_labels) > 0
                
                base_name = f"{file_idx}_slice_{i}"
                if is_positive:
                    save_img_path = pos_img_dir / f"{base_name}.png"
                    save_lbl_path = pos_lbl_dir / f"{base_name}.txt"
                    
                    cv2.imwrite(str(save_img_path), img_slice)
                    with open(save_lbl_path, 'w') as f_lbl:
                        f_lbl.write('\n'.join(slice_labels))
                    
                    viz_candidates.append((save_img_path, save_lbl_path))
                else:
                    save_img_path = neg_img_dir / f"{base_name}.png"
                    cv2.imwrite(str(save_img_path), img_slice)

    # ==========================================
    # 4. 可视化检查 (Visualization)
    # ==========================================
    print("\n正在生成可视化检查样本...")
    
    if len(viz_candidates) > 0:
        # 随机抽取 NUM_VISUAL_SAMPLES 个 (不足 NUM_VISUAL_SAMPLES 个取全部)
        random.seed(RANDOM_SEED)
        viz_samples = random.sample(viz_candidates, min(NUM_VISUAL_SAMPLES, len(viz_candidates)))
        
        for img_p, lbl_p in viz_samples:
            img = cv2.imread(str(img_p))

            if img is None:
                raise ValueError(f"Image not found: {img_p}")

            h, w, _ = img.shape
            
            if lbl_p.exists():
                with open(lbl_p, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    
                    # 反算像素坐标
                    box_w = bw * w
                    box_h = bh * h
                    box_cx = cx * w
                    box_cy = cy * h
                    
                    x1 = int(box_cx - box_w / 2)
                    y1 = int(box_cy - box_h / 2)
                    x2 = int(box_cx + box_w / 2)
                    y2 = int(box_cy + box_h / 2)
                    
                    # 画框 (BGR: 红色)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"Class {cls_id}", (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 保存到 visual 文件夹
            cv2.imwrite(str(vis_dir / f"vis_{img_p.name}"), img)
        print(f"已保存 {len(viz_samples)} 张可视化图片至 {vis_dir}")
    else:
        print("未生成任何正样本，跳过可视化。")

    print(f"\n全部处理完成！数据已保存至: {output_dir}")

if __name__ == "__main__":
    main()