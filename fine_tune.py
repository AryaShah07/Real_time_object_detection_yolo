# -----------------------------------------------------------
# YOLOv8 Hyperparameter Tuning Script
# fine_tune.py
# -----------------------------------------------------------

from ultralytics import YOLO
import torch
import os

def run_tuning():

    device = 0 if torch.cuda.is_available() else 'cpu'
    print("Using device:", "GPU" if device == 0 else "CPU")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(repo_root, "weights", "last.pt")  # keep weights in repo
    data_yaml = os.path.join(repo_root, "data.yaml")

    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"[Warning] Checkpoint not found at {checkpoint_path}. Using pretrained YOLOv8n instead.")
        checkpoint_path = "yolov8n.pt"  # use the smallest model for speed

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    
    # 3. Load YOLOv8 Model
    model = YOLO(checkpoint_path)

    
    # 4. Run Hyperparameter Tuning
    model.tune(
        data=data_yaml,
        epochs=15,            
        batch=16,             
        imgsz=256,           
        optimizer="AdamW",
        device=device,
        save=True,
        project=repo_root,
        name="tuning_results",
        workers=2,           
        mosaic=0.0           
    )

if __name__ == "__main__":
    run_tuning()