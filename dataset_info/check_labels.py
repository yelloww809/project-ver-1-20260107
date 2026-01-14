import os
from pathlib import Path
from tqdm import tqdm

# =================配置区域=================
# 数据集根目录 (你的 v8_small_jpg_1 路径)
DATASET_ROOT = Path(r'E:\huangwenhao\processed_datasets\v8\v8_small_jpg_1')

# 检查报告保存路径
REPORT_PATH = DATASET_ROOT / 'inspection_report.txt'

# 假设图片分辨率 (用于将归一化坐标换算回像素)
# 如果你的切片大小不是 640x640，请修改这里
IMG_WIDTH = 640
IMG_HEIGHT = 640

# [核心阈值] 判定为“异常极小框”的像素阈值
# 如果宽或高小于 3 个像素，就被视为可能导致梯度爆炸的危险框
MIN_PIXEL_THRESHOLD = 0.9

# =========================================

def check_label_file(file_path):
    """
    检查单个 txt 文件，返回异常列表
    """
    issues = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return [] # 空文件通常是负样本，没问题

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            parts = line.split()
            
            # 1. 检查格式是否为 5 列
            if len(parts) != 5:
                issues.append(f"Line {line_idx+1}: 格式错误 (列数!=5) -> {line}")
                continue
                
            cls_id, cx, cy, w, h = parts
            
            try:
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                cls_id = int(cls_id)
            except ValueError:
                issues.append(f"Line {line_idx+1}: 非数值数据 -> {line}")
                continue

            # 2. 检查数值范围 (YOLO 必须在 0-1 之间)
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                issues.append(f"Line {line_idx+1}: 坐标越界 (必须在0-1之间) -> cx={cx}, cy={cy}, w={w}, h={h}")

            # 3. [关键] 检查极小框 (Micro/Sliver Boxes)
            pixel_w = w * IMG_WIDTH
            pixel_h = h * IMG_HEIGHT
            
            is_too_small = False
            error_msg = []
            
            if pixel_w < MIN_PIXEL_THRESHOLD:
                error_msg.append(f"宽仅 {pixel_w:.2f} px")
                is_too_small = True
            
            if pixel_h < MIN_PIXEL_THRESHOLD:
                error_msg.append(f"高仅 {pixel_h:.2f} px")
                is_too_small = True
                
            if is_too_small:
                issues.append(f"Line {line_idx+1}: [极小框警告] {' & '.join(error_msg)} (Norm: w={w:.6f}, h={h:.6f})")

    except Exception as e:
        issues.append(f"文件读取错误: {str(e)}")
        
    return issues

def main():
    print(f"开始检查数据集: {DATASET_ROOT}")
    print(f"判定标准: 宽或高 < {MIN_PIXEL_THRESHOLD} 像素 (基于 {IMG_WIDTH}x{IMG_HEIGHT} 分辨率)")
    
    labels_dir = DATASET_ROOT / 'labels'
    if not labels_dir.exists():
        print(f"错误: 找不到 labels 文件夹: {labels_dir}")
        return

    # 递归查找所有 .txt 文件
    all_txt_files = list(labels_dir.rglob('*.txt'))
    print(f"共发现 {len(all_txt_files)} 个标签文件，开始扫描...")

    total_issues = 0
    files_with_issues = 0
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as report:
        report.write(f"数据集异常检查报告\n")
        report.write(f"时间: {os.path.dirname(__file__)}\n")
        report.write(f"检查路径: {DATASET_ROOT}\n")
        report.write(f"极小框阈值: {MIN_PIXEL_THRESHOLD} 像素\n")
        report.write("="*50 + "\n\n")

        for txt_file in tqdm(all_txt_files):
            # 获取相对路径 (例如 train/123.txt)
            rel_path = txt_file.relative_to(labels_dir)
            file_issues = check_label_file(txt_file)
            
            if file_issues:
                files_with_issues += 1
                total_issues += len(file_issues)
                
                report.write(f"文件: {rel_path}\n")
                for issue in file_issues:
                    report.write(f"  - {issue}\n")
                report.write("-" * 30 + "\n")

        report.write(f"\n{'='*50}\n")
        report.write(f"扫描结束。\n")
        report.write(f"涉及异常文件数: {files_with_issues}\n")
        report.write(f"异常条目总数: {total_issues}\n")

    print(f"\n检查完成！")
    print(f"发现异常文件: {files_with_issues} 个")
    print(f"详情已写入报告: {REPORT_PATH}")
    print("请打开报告查看是否有'极小框警告'。")

if __name__ == "__main__":
    main()