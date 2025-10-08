import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import os
from PIL import Image
import io
import requests

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(
    page_title="YOLO Object Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# ----------------------------
# Flask API Base URL
# ----------------------------
FLASK_API_URL = "http://127.0.0.1:8000"  # Make sure your Flask API is running

# ----------------------------
# Load Model, YAML Config, and Training Results (Optional for Dashboard)
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'runs/detect/final_best_train4/weights/best.pt')
        if os.path.exists(model_path):
            return YOLO(model_path)
        else:
            st.warning(f"Model not found at: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_yaml_config():
    try:
        yaml_path = os.path.join(os.path.dirname(__file__), 'data.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            st.warning(f"data.yaml not found at: {yaml_path}")
        return None
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        return None

@st.cache_data
def load_training_results():
    try:
        results_path = os.path.join(os.path.dirname(__file__), 'runs/detect/final_best_train4/results.csv')
        if os.path.exists(results_path):
            return pd.read_csv(results_path)
        else:
            st.warning(f"Results file not found at: {results_path}")
        return None
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return None

# Load optional data
model = load_model()
config = load_yaml_config()
results_df = load_training_results()

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Model Testing", "Live Video Feed", "Dataset Info"])

# ----------------------------
# Dashboard Page
# ----------------------------
if page == "Dashboard":
    st.title("Training Performance Dashboard")
    
    if results_df is not None:
        tab1, tab2, tab3 = st.tabs(["Loss Metrics", "Performance Metrics", "Learning Rate"])
        
        with tab1:
            st.subheader("Training and Validation Losses")
            st.markdown("""
            **What these metrics mean:**
            - **Box Loss**: How well the model predicts object locations (bounding boxes)
            - **Class Loss**: How well the model identifies object categories
            - Lower values indicate better performance
            """)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['train/box_loss'], 
                                     name='Train Box Loss', line=dict(color='#1f77b4')))
            fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['val/box_loss'], 
                                     name='Val Box Loss', line=dict(color='#1f77b4', dash='dash')))
            fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['train/cls_loss'], 
                                     name='Train Class Loss', line=dict(color='#ff7f0e')))
            fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['val/cls_loss'], 
                                     name='Val Class Loss', line=dict(color='#ff7f0e', dash='dash')))
            fig.update_layout(
                title="Model Training Progress",
                xaxis_title="Training Epoch",
                yaxis_title="Loss Value (lower is better)",
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='white',
                height=500,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Model Performance Metrics")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['metrics/mAP50(B)'], name='mAP50', line=dict(color='#2ca02c')))
            fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['metrics/mAP50-95(B)'], name='mAP50-95', line=dict(color='#d62728')))
            fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['metrics/precision(B)'], name='Precision', line=dict(color='#9467bd')))
            fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['metrics/recall(B)'], name='Recall', line=dict(color='#8c564b')))
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Metric Value",
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='white',
                height=500,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Learning Rate Progress")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df['epoch'], y=results_df['lr/pg0'], name='Learning Rate', line=dict(color='#17becf')))
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Learning Rate",
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='white',
                height=500,
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', type='log')
            )
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Model Testing (Static Image via Flask API)
# ----------------------------
elif page == "Model Testing":
    st.title("Test Model on Images (via Flask API)")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        
    
    if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                # Open image directly from uploaded file safely
                image = Image.open(uploaded_file).convert("RGB")
                image_array = np.array(image)
                
                # Save a copy in outputs folder
                save_path = os.path.join(OUTPUT_FOLDER, uploaded_file.name)
                image.save(save_path)
                
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
                
                # Run YOLO prediction with confidence threshold
                results = model.predict(source=image_array, conf=confidence_threshold, imgsz=640)

                
                st.subheader("Detection Result")
                result_plot = results[0].plot()
                st.image(result_plot, use_container_width=True)
                
                # Collect detections
                detections_list = []
                for res in results:
                    for box in res.boxes:
                        detections_list.append({
                            "class": res.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "xmin": int(box.xyxy[0][0]),
                            "ymin": int(box.xyxy[0][1]),
                            "xmax": int(box.xyxy[0][2]),
                            "ymax": int(box.xyxy[0][3])
                        })
                
                # Save CSV in outputs folder
                if detections_list:
                    df = pd.DataFrame(detections_list)
                    csv_filename = f"predictions_{uploaded_file.name.rsplit('.',1)[0]}.csv"
                    csv_path = os.path.join(OUTPUT_FOLDER, csv_filename)
                    df.to_csv(csv_path, index=False)
                    
                    st.subheader("Download Predictions")
                    st.download_button(
                        label="Download CSV",
                        data=open(csv_path, "rb"),
                        file_name=csv_filename,
                        mime="text/csv"
                    )
                    st.success(f"CSV saved in outputs folder: {csv_filename}")
                
            except Exception as e:
                st.error(f"Error reading uploaded image: {str(e)}")

        try:
            response = requests.post(f"{FLASK_API_URL}/detect", files=files)
            if response.status_code == 200:
                data = response.json()
                annotated_url = FLASK_API_URL + data["annotated_image"]

                # Fetch annotated image correctly
                img_response = requests.get(annotated_url)
                if img_response.status_code == 200:
                    annotated_img = Image.open(io.BytesIO(img_response.content))
                    st.subheader("Detection Result")
                    st.image(annotated_img, use_container_width=True)  # ✅ fixed
                else:
                    st.error(f"Failed to fetch annotated image: {img_response.status_code}")

                st.subheader("Detections")
                for det in data["detections"]:
                    st.write(f"Class: {det['class']} | Confidence: {det['confidence']:.2f}")
            else:
                st.error(f"Error from Flask API: {response.json().get('error')}")
        except Exception as e:
            st.error(f"Could not connect to Flask API: {e}")

# ----------------------------
# Live Video Feed (Dynamic Input)
# ----------------------------
elif page == "Live Video Feed":
    st.title("Live Video Feed (via Flask API)")

    st.markdown(
        """
        <h5>Real-time webcam detection via Flask API</h5>
        <img src="http://127.0.0.1:8000/live" width="700" />
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Dataset Info
# ----------------------------
elif page == "Dataset Info":
    st.title("Dataset Information")
    st.subheader("Classes")
    if 'names' in config:
        classes_df = pd.DataFrame({'Class Name': config['names']})
        classes_df.index = range(len(classes_df))
        st.dataframe(classes_df)
    
    st.subheader("Dataset Structure")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Training Dataset")
        train_path = os.path.join(os.path.dirname(__file__), 'train')
        if os.path.exists(train_path):
            n_train_images = len(os.listdir(os.path.join(train_path, 'images')))
            n_train_labels = len(os.listdir(os.path.join(train_path, 'labels')))
            st.write(f"Images: {n_train_images}")
            st.write(f"Labels: {n_train_labels}")
    with col2:
        st.write("Validation Dataset")
        valid_path = os.path.join(os.path.dirname(__file__), 'valid')
        if os.path.exists(valid_path):
            n_valid_images = len(os.listdir(os.path.join(valid_path, 'images')))
            n_valid_labels = len(os.listdir(os.path.join(valid_path, 'labels')))
            st.write(f"Images: {n_valid_images}")
            st.write(f"Labels: {n_valid_labels}")

# ----------------------------
# Footer
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Created with Streamlit")