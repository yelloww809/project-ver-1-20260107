from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\pretrained_model\yolo11\yolo11m.pt")

    model.train(
        data=r"E:\huangwenhao\processed_datasets\yolo_train_set_v6_3000_large_fix_res\yolo_train_set_v6_3000_large_fix_res.yaml",
        epochs=200,
        patience=100,
        imgsz=800,
        batch=32,
        cos_lr=True,
        device='cuda:0',
        project=r'E:\huangwenhao\runs',
        name='train_v6_3000_large_fix_res_imgsz800',
    )

if __name__ == '__main__':
    main()