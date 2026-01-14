from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\pretrained_model\yolo11\yolo11m.pt")

    model.train(
        data=r"E:\huangwenhao\processed_datasets\v8\v8_small_jpg_1\v8_small_jpg_1.yaml",
        epochs=40,
        patience=20,
        imgsz=640,
        batch=32,
        cos_lr=True,
        device='cuda:0',
        project=r'E:\huangwenhao\runs\v8\train', 
        name='train_v8_small_jpg_1_epochs40',
    )

if __name__ == '__main__':
    main()