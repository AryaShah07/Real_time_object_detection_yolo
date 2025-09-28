Real-Time Object Detection using YOLOv8
This project is about detecting objects in real time with YOLOv8. The model can take input from images, videos, or even a webcam and highlight the detected objects with bounding boxes.

✨ Features
Train the model on a custom dataset

Run detection on images, videos, or live camera

Save the results with bounding boxes and labels

Check accuracy and performance using evaluation metrics

❓ Why YOLOv8?
We decided to use YOLOv8 because:

It gives good accuracy while being fast

It is easy to train and test in one framework

It has many tutorials and documentation available

It supports transfer learning with pretrained weights

(We are aware that YOLOv11 is newer and gives better results, but YOLOv8 is more stable, easier to use, and well supported right now. So we selected YOLOv8 for this project.)

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
We analyzed different YOLOv8 pre-trained models based on size, speed, accuracy, and hardware requirements:

YOLOv8-n (Nano)

Extremely lightweight, fastest inference & training speed

Runs efficiently on CPUs or low-end GPUs

Slightly lower accuracy but good enough for real-time apps

✅ Best when speed and low resource usage are critical

YOLOv8-s (Small)

Better accuracy than Nano

Moderate inference speed, suitable for standard GPUs

Balanced choice between speed & accuracy

YOLOv8-m, YOLOv8-l, YOLOv8-x

Higher accuracy, but much heavier and slower

Require powerful GPUs

Selection Justification
For this project, the focus is real-time object detection with limited hardware resources.
We selected YOLOv8n because it provides fast inference, low memory usage, and acceptable accuracy, which aligns perfectly with our project goals.

