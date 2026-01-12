from ultralytics import YOLO

def main():
    model = YOLO("/data/hwh_data_folder/models/yolo11n.pt")

    model.train(
        data="/data/hwh_data_folder/processed_datasets/yolo_train_set_v5_500_small/yolo_train_set_v5_500_small.yaml",
        epochs=10,
        imgsz=640,
        batch=4,
        cos_lr=True,
        device='cuda:1',
        project='/data/hwh_data_folder/runs',
        name='train_test_small_1',
    )

if __name__ == '__main__':
    main()