from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\pretrained_model\yolo11\yolo11m.pt")

    model.train(
        data=r"E:\huangwenhao\processed_datasets\v7_large_jpg_1\v7_large_jpg_1.yaml",
        epochs=200,
        patience=100,
        imgsz=640,
        batch=32,
        cos_lr=True,
        device='cuda:0',
        project=r'E:\huangwenhao\runs',
        name='train_v7_large_jpg_1',
    )

if __name__ == '__main__':
    main()