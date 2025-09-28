Real-Time Object Detection using YOLOv8
This project detects objects in real time using YOLOv8. The model can take input from images, videos, or a webcam and highlight detected objects with bounding boxes.

✨ Features
Train the model on a custom dataset

Run detection on images, videos, or live camera

Save results with bounding boxes and labels

Check accuracy and performance using evaluation metrics

❓ Why YOLOv8?
We decided to use YOLOv8 because:

Good accuracy while being fast

Easy to train and test in one framework

Extensive tutorials and documentation available

Supports transfer learning with pretrained weights

Note: YOLOv11 is newer and may give better results, but YOLOv8 is more stable, easier to use, and well supported, so we selected it for this project.

📦 Required Python Libraries
Library	Purpose
ultralytics	Main YOLOv8 library
torch, torchvision	Deep learning framework (PyTorch)
opencv-python	Image/video handling
numpy	Array and math operations
pandas	Working with dataset files
matplotlib	Visualizing results and training metrics
seaborn	Extra plots (like confusion matrix)
pyyaml	Reading dataset configuration files
tqdm	Progress bar during training
scikit-learn	Splitting dataset, evaluation (optional)
scipy	Numerical functions (optional)
tensorboard	Monitoring training progress (optional)
roboflow	Dataset import (optional)
psutil	Monitor CPU/GPU usage (optional)
🔍 Research & Select YOLOv8 Variant
Model	Description	Use-case
YOLOv8-n (Nano)	Extremely lightweight, fastest inference & training	Best when speed & low resource usage are critical
YOLOv8-s (Small)	Better accuracy than Nano, moderate inference speed	Balanced choice between speed & accuracy
YOLOv8-m / l / x	Higher accuracy, heavier & slower	Requires powerful GPUs
Selection Justification:
We focused on real-time object detection with limited hardware resources. YOLOv8n provides fast inference, low memory usage, and acceptable accuracy, aligning perfectly with our project goals.

📊 Performance Metrics & Modeling Approach
Evaluation Metrics
mAP@50: Detection accuracy at IoU threshold 0.5

mAP@50-95: Mean Average Precision across IoU thresholds (0.5–0.95)

Precision: Correctness of predicted detections

Recall: Ability to detect all relevant objects

Losses (Box, Cls, DFL): Indicators of localization, classification, and distribution errors

Results Summary (Epochs 1–28)
Loss Reduction: Training and validation losses decreased steadily

Training Box Loss: 2.01 → 1.58

Training Cls Loss: 4.23 → 2.59

Training DFL Loss: 1.89 → 1.58

Validation losses decreased similarly, showing strong learning and reduced overfitting

Precision: Improved from 0.28 → ~0.40, indicating more confident predictions

Recall: Increased from 0.03 → 0.14, showing the model detects more true objects over time

mAP@50: Increased from 0.002 → 0.108

mAP@50-95: Improved from 0.0009 → 0.066

✅ Key Takeaway: The model demonstrates consistent learning, with reduced losses, improved precision, and gradually increasing recall and mAP. Trends indicate that further training, data augmentation, and hyperparameter tuning will enhance detection performance.

Final Modeling Choice
YOLOv8n (Nano): ✅ Chosen for efficiency, speed, and low hardware requirements

YOLOv8s (Small): Slightly higher accuracy but heavier, not ideal for resource-constrained hardware

Final Decision:
We use YOLOv8n because it enables real-time inference on limited hardware while showing stable and progressive accuracy growth.