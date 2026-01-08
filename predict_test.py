from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
model.predict(
    source='/data/hwh_data_folder/project-ver-1-20260107/ultralytics/assets',
    save=True,
    show=False
)