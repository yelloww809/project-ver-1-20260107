import cv2
import os

def visualize_yolo(img_path, txt_path, output_path='visualized_result.png'):
    # 1. 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"错误: 找不到图片文件: {img_path}")
        return
    if not os.path.exists(txt_path):
        print(f"错误: 找不到标签文件: {txt_path}")
        return

    # 2. 读取图片
    # cv2.imread 默认加载为 BGR 格式
    img = cv2.imread(img_path)
    if img is None:
        print("错误: 图片无法读取 (可能是文件损坏)")
        return

    # 获取图片尺寸 (用于反归一化)
    height, width, _ = img.shape
    print(f"图片加载成功: {width}x{height}")

    # 3. 读取标签文件
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    print(f"找到 {len(lines)} 个目标")

    # 4. 遍历每一行标签并画框
    for line in lines:
        data = line.strip().split()
        if len(data) < 5:
            continue
            
        # 解析 YOLO 格式: class_id, x_center, y_center, w, h
        class_id = int(data[0])
        x_center = float(data[1])
        y_center = float(data[2])
        w = float(data[3])
        h = float(data[4])

        # --- 核心步骤: 将归一化坐标 (0-1) 转换为 像素坐标 ---
        # 计算中心点的像素坐标
        x_c_pixel = x_center * width
        y_c_pixel = y_center * height
        
        # 计算宽高的像素长度
        w_pixel = w * width
        h_pixel = h * height

        # 计算左上角 (x1, y1) 和 右下角 (x2, y2) 坐标
        x1 = int(x_c_pixel - w_pixel / 2)
        y1 = int(y_c_pixel - h_pixel / 2)
        x2 = int(x_c_pixel + w_pixel / 2)
        y2 = int(y_c_pixel + h_pixel / 2)

        # 5. 在图上画矩形框
        # 参数: 图片, 左上角, 右下角, 颜色(BGR, 绿色), 线宽
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # (可选) 添加类别标签文字
        text = f"Class {class_id}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 6. 保存结果图片
    cv2.imwrite(output_path, img)
    print(f"✅ 处理完成！结果已保存为: {os.path.abspath(output_path)}")

# --- 配置路径 ---
image_file = '/data/hwh_data_folder/processed_datasets/yolo_training_set_v2/images/positive/857_slice_11.png'
label_file = '/data/hwh_data_folder/processed_datasets/yolo_training_set_v2/labels/positive/857_slice_11.txt'

# --- 运行 ---
visualize_yolo(image_file, label_file)