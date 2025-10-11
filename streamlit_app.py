import streamlit as st
import pandas as pd
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

# --- NEW: outputs folder and small helpers ---
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@st.cache_data
def summarize_results(results_df):
    # returns small KPI dict used in dashboard
    if results_df is None or results_df.empty:
        return {}
    best_map = float(results_df['metrics/mAP50(B)'].max())
    last = results_df.iloc[-1]
    last_map = float(last['metrics/mAP50(B)'])
    final_train_loss = float(last['train/box_loss'])
    final_val_loss = float(last['val/box_loss'])
    total_time = float(results_df['time'].sum())
    return {
        "best_map": best_map,
        "last_map": last_map,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "total_time_s": total_time
    }

def predict_image(image_array, model, conf=0.25, iou=0.45, imgsz=640):
    # Returns annotated image (numpy/PIL) and detections list
    if model is None:
        raise RuntimeError("Model not loaded")
    results = model.predict(source=image_array, conf=conf, iou=iou, imgsz=imgsz)
    annotated = results[0].plot()  # numpy image with annotations
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
    return annotated, detections_list

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

# --- NEW: Small CSS / hero banner to apply two-color aesthetic (primary + background) ---
# Use same colors as .streamlit/config.toml and add a secondary accent
primary = "#0B5FFF"            # primary accent
background = "#F7F9FC"         # app background
secondary = "#FFFFFF"          # card / secondary background (clean white)
text_color = "#0B254A"         # primary text color
app_font = "sans-serif"        # single font family enforced

st.markdown(
    f"""
    <style>
    /* Enforce single font app-wide */
    html, body, [class*="css"]  {{ font-family: {app_font} !important; color: {text_color}; }}

    /* Hero banner */
    .hero {{
        padding: 18px;
        border-radius: 10px;
        background: linear-gradient(90deg, {primary}22, {background});
        color: {text_color};
        margin-bottom: 18px;
    }}
    .hero h2 {{ margin:4px 0; color: {primary}; font-weight:700; }}

    /* Quick action cards */
    .quick-card {{
        border-radius: 8px;
        padding: 12px;
        background: {secondary};
        box-shadow: 0 6px 18px rgba(11,95,255,0.06);
        border: 1px solid rgba(11,95,255,0.06);
        margin-bottom: 8px;
    }}

    /* Gallery grid and items */
    .gallery-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 8px;
    }}
    .gallery-item img {{ width: 100%; border-radius: 6px; box-shadow: 0 4px 10px rgba(11,95,255,0.04); }}

    /* Buttons (Streamlit default buttons keep native styling; add subtle focus) */
    .stButton>button:focus {{ outline-color: {primary}; box-shadow: 0 0 0 3px rgba(11,95,255,0.08); }}

    /* Small utility */
    .muted {{ color: rgba(11,37,74,0.6); font-size: 0.95rem; }}
    </style>
    <div class="hero">
        <h2>YOLO Dashboard</h2>
        <div class="muted">Clean • Focused • Production-ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Model Testing", "Live Video Feed", "Dataset Info"])

# --- NEW: Model management UI (top of the app, under header) ---
# place this near the top after loading model/config/results_df
st.sidebar.markdown("## Model Management")
model_path_input = st.sidebar.text_input("Model weights path", value=os.path.join(os.path.dirname(__file__), 'runs/detect/final_best_train4/weights/best.pt'))
if st.sidebar.button("Reload model"):
    try:
        model = YOLO(model_path_input) if os.path.exists(model_path_input) else None
        st.sidebar.success("Model reloaded" if model is not None else "Model path invalid")
    except Exception as e:
        st.sidebar.error(f"Reload failed: {e}")

st.sidebar.markdown("---")
# quick model status
model_status = "Loaded" if model is not None else "Not loaded"
st.sidebar.write(f"Model status: **{model_status}**")
st.sidebar.write(f"Outputs: {OUTPUT_FOLDER}")

# ----------------------------
# Dashboard Page
# ----------------------------
if page == "Dashboard":
    st.title("Training Performance Dashboard")
    # --- NEW: KPI cards ---
    if results_df is not None:
        kpi = summarize_results(results_df)
        if kpi:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Best mAP50", f"{kpi['best_map']:.4f}")
            col2.metric("Last epoch mAP50", f"{kpi['last_map']:.4f}")
            col3.metric("Final train box loss", f"{kpi['final_train_loss']:.4f}")
            col4.metric("Final val box loss", f"{kpi['final_val_loss']:.4f}")
            st.caption(f"Total training time ≈ {kpi['total_time_s']:.0f}s")
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
    st.title("Test Model on Images")
    st.markdown("Upload one or more images, set thresholds, then click 'Run Predictions'.")

    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("NMS IoU threshold", 0.0, 1.0, 0.45, 0.01)
    imgsz = st.selectbox("Inference image size", options=[320, 416, 640, 1024], index=2)
    multiple = st.checkbox("Allow multiple uploads", value=True)

    uploaded_files = st.file_uploader("Choose image(s)...", type=['jpg','jpeg','png'], accept_multiple_files=multiple)
    run_btn = st.button("Run Predictions")

    if run_btn and uploaded_files:
        if model is None:
            st.error("Model not loaded. Use Model Management to load weights.")
        else:
            progress = st.progress(0)
            total = len(uploaded_files)
            all_dets = []
            for idx, uf in enumerate(uploaded_files, start=1):
                try:
                    uf.seek(0)
                    img = Image.open(uf).convert("RGB")
                    arr = np.array(img)
                    with st.spinner(f"Running inference on {uf.name} ({idx}/{total})..."):
                        annotated, detections = predict_image(arr, model, conf=conf, iou=iou, imgsz=imgsz)
                    out_name = f"annotated_{os.path.basename(uf.name)}"
                    out_path = os.path.join(OUTPUT_FOLDER, out_name)
                    # save annotated image (OpenCV expects BGR)
                    if isinstance(annotated, np.ndarray):
                        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                    else:
                        annotated.save(out_path)
                    # display results
                    st.subheader(f"Result: {uf.name}")
                    st.image(annotated, use_column_width=True)
                    if detections:
                        df = pd.DataFrame(detections)
                        st.dataframe(df)
                        # save csv
                        csv_name = f"{os.path.splitext(uf.name)[0]}_predictions.csv"
                        csv_path = os.path.join(OUTPUT_FOLDER, csv_name)
                        df.to_csv(csv_path, index=False)
                        st.download_button("Download CSV", data=open(csv_path, "rb"), file_name=csv_name, mime="text/csv")
                        all_dets.append(df)
                    else:
                        st.info("No detections for this image.")
                except Exception as e:
                    st.error(f"Failed on {uf.name}: {e}")
                progress.progress(int(idx/total*100))
            progress.empty()
            st.success("Batch predictions completed.")
            # aggregated CSV if multiple
            if len(all_dets) > 1:
                agg = pd.concat(all_dets, ignore_index=True)
                agg_path = os.path.join(OUTPUT_FOLDER, "predictions_aggregated.csv")
                agg.to_csv(agg_path, index=False)
                st.download_button("Download aggregated CSV", data=open(agg_path, "rb"), file_name="predictions_aggregated.csv", mime="text/csv")

    # Backwards-compatible single-file immediate mode (if user just uploaded and didn't click run)
    elif uploaded_files and not run_btn:
        st.info("Set parameters and click 'Run Predictions' to process uploaded images.")

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