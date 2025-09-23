from ultralytics import YOLO
import torch
import os
import yaml

def run_training():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print("Using device:", "GPU" if device == 0 else "CPU")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(repo_root, "weights", "last.pt")
    data_yaml = os.path.join(repo_root, "data.yaml")
    hyp_yaml = os.path.join(repo_root, "tuning_results", "best_hyperparameters.yaml")

    # Use last.pt if exists, else fallback to yolov8n.pt
    if not os.path.exists(checkpoint_path):
        print(f"[Warning] Checkpoint not found at {checkpoint_path}. Using yolov8n.pt instead.")
        checkpoint_path = "yolov8n.pt"

    # Load best hyperparameters
    with open(hyp_yaml, "r") as f:
        hyp = yaml.safe_load(f)

    # Remove comments and non-hyperparameter lines if present
    hyp = {k: v for k, v in hyp.items() if not k.startswith("#")}

    # Fix: Ensure close_mosaic is int if present
    if "close_mosaic" in hyp:
        hyp["close_mosaic"] = int(hyp["close_mosaic"])

    model = YOLO(checkpoint_path)

    model.train(
        data=data_yaml,
        epochs=50,  # set as needed for final training
        batch=16,   # adjust as per your GPU RAM
        imgsz=640,  # use larger size for best results if possible
        device=device,
        optimizer="AdamW",
        name="final_best_train",
        **hyp
    )

if __name__ == "__main__":
    run_training()
