from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\results\v9\train\train_v9_large_2_1\weights\best.pt")

    model.val(
        data=r"E:\huangwenhao\processed_datasets\v9\v9_large_2\v9_large_2.yaml",
        split='test',
        imgsz=640,
        batch=32,
        device='cuda:1',
        project=r'E:\huangwenhao\results\v9\test',
        name='test_v9_large_2_1_conf0.001',
        conf=0.001,
        iou=0.7,
    )

if __name__ == '__main__':
    main()