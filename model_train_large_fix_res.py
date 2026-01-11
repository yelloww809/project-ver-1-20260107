from ultralytics import YOLO

def main():
    model = YOLO("/data/hwh_data_folder/models/yolo11m.pt")

    model.train(
        data="/data/hwh_data_folder/processed_datasets/yolo_train_set_v6_3000_large_fix_res/yolo_train_set_v6_3000_large_fix_res.yaml",
        epochs=200,
        patience=100,
        imgsz=640,
        batch=16,
        cos_lr=True,
        device='cuda:1',
        project='/data/hwh_data_folder/runs',
        name='train_v6_3000_large_fix_res',
    )

if __name__ == '__main__':
    main()