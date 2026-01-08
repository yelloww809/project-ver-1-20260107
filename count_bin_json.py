from pathlib import Path

def count_files(directory_path):
    # 将字符串路径转换为 Path 对象
    p = Path(directory_path)
    
    # 检查路径是否存在
    if not p.exists():
        print(f"错误: 路径 '{directory_path}' 不存在。")
        return

    bin_count = 0
    json_count = 0
    
    print(f"正在统计目录: {directory_path} ...")

    # iterdir() 遍历当前目录下的所有内容（不包含子目录）
    for file in p.iterdir():
        if file.is_file():
            # suffix 获取文件后缀名
            if file.suffix == '.bin':
                bin_count += 1
            elif file.suffix == '.json':
                json_count += 1
                
    print("-" * 30)
    print(f"统计结果:")
    print(f"🔹 .bin 文件数量:  {bin_count}")
    print(f"🔸 .json 文件数量: {json_count}")
    print("-" * 30)

# 执行统计
target_path = "/data/hwh_data_folder/dataset/test_public"
count_files(target_path)