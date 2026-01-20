from ultralytics import YOLO

def main():
    model = YOLO(r"E:\huangwenhao\runs\v8\train_v8_large_jpg_2\weights\best.pt")

    model.val(
        data=r"E:\huangwenhao\processed_datasets\v8\v8_large_jpg_2\v8_large_jpg_2.yaml",
        split='test',
        imgsz=640,
        batch=32,
        device='cuda:1',
        project=r'E:\huangwenhao\runs\v8\test',
        name='test_v8_large_jpg_2_epochs160'
    )

if __name__ == '__main__':
    main()