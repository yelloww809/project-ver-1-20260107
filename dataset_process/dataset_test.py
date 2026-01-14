import os
import shutil
import random
import json
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 (必须与训练预处理脚本完全一致) =================
# 1. 随机种子 (核心！)
RANDOM_SEED = 42

# 2. 原始数据集路径 (存放所有 .bin 和 .json 的地方)
SOURCE_DIR = Path(r"E:\huangwenhao\dataset\train")

# 3. 目标输出路径 (Test集存放位置)
DEST_DIR = Path(r"E:\huangwenhao\processed_datasets\dataset_test")

# 4. 切分比例 (必须与当时训练时的比例一致！)
# 假设当时是 80% 训练, 10% 验证, 10% 测试
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 1 - 0.8 - 0.1 = 0.1

# =======================================================================

def extract_test_set():
    print(f">>> 正在初始化...")
    print(f"    源目录: {SOURCE_DIR}")
    print(f"    目标目录: {DEST_DIR}")
    print(f"    随机种子: {RANDOM_SEED}")
    
    # 0. 准备目标目录
    if not DEST_DIR.exists():
        DEST_DIR.mkdir(parents=True)
        print(f"    [Info] 创建目标目录: {DEST_DIR}")
    else:
        print(f"    [Warning] 目标目录已存在，新文件将覆盖旧文件。")

    # 1. 获取所有文件并排序 (关键步骤！)
    # 必须先 sorted，因为不同系统 glob 返回的顺序可能不同，不排序直接 shuffle 结果会乱
    all_bin_files = sorted(list(SOURCE_DIR.glob('*.bin')))
    total_files = len(all_bin_files)
    
    if total_files == 0:
        print(f"[Error] 源目录下没有找到 .bin 文件！请检查路径。")
        return

    print(f"    找到了 {total_files} 个样本文件。")

    # 2. 打乱文件 (复现当时的随机状态)
    random.seed(RANDOM_SEED)
    random.shuffle(all_bin_files)

    # 3. 计算索引切分点
    num_train = int(total_files * TRAIN_RATIO)
    num_val = int(total_files * VAL_RATIO)
    
    # 4. 提取 Test 集切片
    # 训练集: [0 : num_train]
    # 验证集: [num_train : num_train + num_val]
    # 测试集: [num_train + num_val : 最后]
    test_files = all_bin_files[num_train + num_val : ]
    
    print(f"    --------------------------------")
    print(f"    [Split Info]")
    print(f"    Train样本数: {num_train}")
    print(f"    Val  样本数: {num_val}")
    print(f"    Test 样本数: {len(test_files)}  <-- 本次将提取这些")
    print(f"    --------------------------------")

    # 5. 开始复制
    print(f">>> 开始复制文件...")
    copy_count = 0
    
    for bin_src in tqdm(test_files, desc="Copying Test Set"):
        # 构造 JSON 路径
        json_src = bin_src.with_suffix('.json')
        
        # 构造目标路径
        bin_dst = DEST_DIR / bin_src.name
        json_dst = DEST_DIR / json_src.name
        
        # 复制 .bin
        shutil.copy2(bin_src, bin_dst)
        
        # 复制 .json (如果存在)
        if json_src.exists():
            shutil.copy2(json_src, json_dst)
        else:
            print(f"[Warning] 找不到对应的 JSON 文件: {json_src.name}")
            
        copy_count += 1

    print(f"\n>>> 提取完成！")
    print(f"    共复制了 {copy_count} 组样本 (.bin + .json) 到 {DEST_DIR}")
    print(f"    现在你可以使用 evaluate_system.py 指向这个文件夹进行测试了。")

if __name__ == "__main__":
    extract_test_set()