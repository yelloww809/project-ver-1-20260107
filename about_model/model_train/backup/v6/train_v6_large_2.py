from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\pretrained_model\yolo11\yolo11m.pt")

    model.train(
        data=r"E:\huangwenhao\processed_datasets\v6_large_2\v6_large_2.yaml",
        epochs=200,
        patience=100,
        imgsz=640,
        batch=32,
        cos_lr=True,
        device='cuda:1',
        project=r'E:\huangwenhao\runs',
        name='train_v6_large_2',
    )

if __name__ == '__main__':
    main()