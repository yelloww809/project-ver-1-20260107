import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

# ================= 配置区域 =================
# 请在这里填写你要统计的文件夹绝对路径
# 注意：Windows 路径建议在引号前加 r，或者使用反斜杠 /

# TARGET_PATH = "/data/hwh_data_folder/processed_datasets/yolo_train_set_v3_500_small/images/train"  
# TARGET_PATH = "/data/hwh_data_folder/processed_datasets/yolo_train_set_v4_500_small/images/train"  

# TARGET_PATH = "/data/hwh_data_folder/processed_datasets/yolo_train_set_v6_3000_large/images/train"  
# TARGET_PATH = "/data/hwh_data_folder/processed_datasets/yolo_train_set_v6_3000_large/images/valid"  

# TARGET_PATH = "/data/hwh_data_folder/processed_datasets/yolo_train_set_v6_3000_large_fix_res/images/train"  
TARGET_PATH = "/data/hwh_data_folder/processed_datasets/yolo_train_set_v6_3000_large_fix_res/images/valid"  

# 结果保存的文件名
OUTPUT_FILENAME = "文件统计报告.txt"
# ===========================================

def generate_report(directory_path, output_file):
    path = Path(directory_path)

    # 1. 校验路径
    if not path.exists():
        return f"❌ 错误：路径 '{directory_path}' 不存在。"
    if not path.is_dir():
        return f"❌ 错误：路径 '{directory_path}' 是一个文件，不是文件夹。"

    # 2. 初始化计数
    file_count = 0
    extension_counter = Counter()

    try:
        # 3. 遍历统计 (仅统计当前层级，不包含子文件夹)
        for item in path.iterdir():
            if item.is_file():
                file_count += 1
                # 获取后缀名，统一转小写，若无后缀则标记
                ext = item.suffix.lower() if item.suffix else "无后缀文件"
                extension_counter[ext] += 1
                
    except PermissionError:
        return "❌ 错误：没有权限访问该文件夹。"

    # 4. 构建报告内容字符串
    lines = []
    lines.append(f"📁 文件夹统计报告")
    lines.append(f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"目标路径: {path.absolute()}")
    lines.append("-" * 40)
    lines.append(f"总文件数量: {file_count}")
    lines.append("-" * 40)

    if file_count == 0:
        lines.append("该文件夹下没有文件。")
    else:
        # 表头
        lines.append(f"{'文件类型':<15} | {'数量':<5}")
        lines.append("-" * 25)
        # 排序输出
        for ext, count in extension_counter.most_common():
            lines.append(f"{ext:<15} | {count:<5}")
    
    lines.append("-" * 40)
    
    return "\n".join(lines)

if __name__ == "__main__":
    print(f"正在扫描: {TARGET_PATH} ...")
    
    # 获取统计报告内容
    report_content = generate_report(TARGET_PATH, OUTPUT_FILENAME)

    print("-" * 20) 
    print(report_content) # 同时在终端打印一遍供预览
    
    # # 保存到 TXT 文件
    # try:
    #     with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
    #         f.write(report_content)
        
    #     print("✅ 统计完成！")
    #     print(f"📄 结果已保存至当前目录下的文件: {OUTPUT_FILENAME}")
        
    # except Exception as e:
    #     print(f"❌ 保存文件失败: {e}")