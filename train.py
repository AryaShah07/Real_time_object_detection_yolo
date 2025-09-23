from ultralytics import YOLO
import torch

def run_training():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print("Using device:", "GPU" if device == 0 else "CPU")

    model = YOLO("yolov8m.pt")

    model.train(
        data="data.yaml",
        epochs=30,               # fewer epochs
        batch=16,                # smaller batch (fits 4GB VRAM better)
        imgsz=416,               # smaller image size for speed
        name="yolov8_fast_train",
        lr0=0.002,
        optimizer="AdamW",
        device=device,
        workers=0,               # Windows fix
        patience=10,             # early stopping
        cache=True,              # cache dataset in memory
        half=True                # use FP16 for faster training
    )

if __name__ == "__main__":
    run_training()
