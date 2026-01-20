from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\pretrained_model\yolo11\yolo11m.pt")

    model.train(
        data=r"E:\huangwenhao\processed_datasets\yolo_train_set_v6_3000_large\yolo_train_set_v6_3000_large.yaml",
        epochs=200,
        patience=100,
        imgsz=640,
        batch=32,
        cos_lr=True,
        device='cuda:1',
        project=r'E:\huangwenhao\runs',
        name='train_v6_3000_large',
    )

if __name__ == '__main__':
    main()