from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\pretrained_model\yolo11\yolo11n.pt")

    model.train(
        data=r"E:\huangwenhao\processed_datasets\v9\v9_base_8\v9_base_8.yaml",
        epochs=100,
        patience=30,
        imgsz=640,
        batch=32,
        # cos_lr=True,
        device='cuda:1',
        project=r'E:\huangwenhao\results\v9\train',
        name='train_v9_base_8_1',
    )

if __name__ == '__main__':
    main()