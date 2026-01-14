from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\runs\v8\train\train_v8_small_jpg_test_3_epochs40\weights\best.pt")

    model.val(
        data=r"E:\huangwenhao\processed_datasets\v8\v8_small_jpg_test_3\v8_small_jpg_3_test.yaml",
        split='test',
        imgsz=640,
        batch=32,
        device='cuda:1',
        project=r'E:\huangwenhao\runs\v8\test',
        name='test_v8_small_jpg_test_3_epochs40'
    )

if __name__ == '__main__':
    main()