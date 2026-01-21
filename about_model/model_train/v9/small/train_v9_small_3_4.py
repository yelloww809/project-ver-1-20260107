from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\pretrained_model\yolo11\yolo11n.pt")

    model.train(
        data=r"E:\huangwenhao\processed_datasets\v9\v9_small_3\v9_small_3.yaml",
        epochs=50,
        patience=20,
        imgsz=640,
        batch=32,
        cos_lr=True,
        device='cuda:1',
        project=r'E:\huangwenhao\results\v9\train',
        name='train_v9_small_3_4',
    )

if __name__ == '__main__':
    main()