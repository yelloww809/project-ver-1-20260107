from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\runs\v8\train\train_v8_small_jpg_test_1_epochs40\weights\best.pt")

    model.val(
        data=r"E:\huangwenhao\processed_datasets\v8\v8_small_jpg_test_1\v8_small_jpg_1_test.yaml",
        split='test',
        imgsz=640,
        batch=32,
        device='cuda:0',
        project=r'E:\huangwenhao\runs\v8\test',
        name='test_v8_small_jpg_test_1_epochs40'
    )

if __name__ == '__main__':
    main()