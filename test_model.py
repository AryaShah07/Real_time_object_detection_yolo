from ultralytics import YOLO
import cv2
import os
import sys

# Define model path - update this to your actual weights location
model_path = 'runs\detect\final_best_train4\weights\best.pt'

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Error: Model weights not found at {os.path.abspath(model_path)}")
    print("Please make sure you have trained the model and the weights file exists.")
    sys.exit(1)

try:
    # Load the trained model
    model = YOLO(model_path)

    # Path to the demo image
    image_path = 'demo.jpg'

    # Perform prediction
    results = model(image_path)

    # Process and save results
    for result in results:
        # Get the plotted image with predictions
        plot = result.plot()

        # Save the output image
        output_path = 'demo_output4.jpg'
        cv2.imwrite(output_path, plot)

    print(f"Prediction completed. Output saved as 'demo_output4.jpg'")

    # Display some basic prediction information
    for result in results:
        print("\nDetection Results:")
        print(f"Number of objects detected: {len(result.boxes)}")
        for box in result.boxes:
            print(f"Class: {result.names[int(box.cls[0])]} | Confidence: {box.conf[0]:.2f}")

except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)
