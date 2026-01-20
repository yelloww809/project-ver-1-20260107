from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
model.predict(
    source=r'E:\huangwenhao\project-ver-1-20260107\ultralytics\assets',
    save=True,
    show=False
)