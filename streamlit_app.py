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
import time
import base64
import json

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(
    page_title="YOLO Object Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# ----------------------------
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
        # primary new default model path (user-provided)
        model_path = os.path.join(os.path.dirname(__file__), 'runs/detect/best_train_results2/weights/best.pt')
        # fallback to previous location if needed
        if not os.path.exists(model_path):
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
        # prefer results from the new run folder, fallback to older run
        results_path = os.path.join(os.path.dirname(__file__), 'runs/detect/best_train_results2/results.csv')
        if not os.path.exists(results_path):
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

# --- THEME / STYLING (replaces prior CSS hero block) ---
# Use a consistent dark palette and a tertiary accent for sidebar cards.
primary = "#0B5FFF"            # main accent
accent = "#00D1B2"             # secondary accent (teal) for subtle highlights
bg = "#071018"                 # app background (dark navy)
panel = "#0F2433"              # panel/card background
card = "#0b1f2a"               # inner card background
text_color = "#E6F0FF"         # primary text color
muted = "rgba(230,240,255,0.72)"  # muted text
app_font = "sans-serif"

st.markdown(
    f"""
    <style>
    /* Apply app-wide font and background */
    html, body, [class*="css"] {{
        font-family: {app_font} !important;
        color: {text_color};
        background-color: {bg} !important;
    }}

    /* Main app area: soften edges */
    .stApp > div {{
        background: linear-gradient(180deg, {bg}, #071827) !important;
        padding-top: 8px;
    }}

    /* Hero banner */
    .hero {{
        padding: 16px 18px;
        border-radius: 10px;
        background: linear-gradient(90deg, rgba(11,95,255,0.08), rgba(0,209,178,0.03));
        color: {text_color};
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.03);
    }}
    .hero h2 {{ margin:4px 0; color: {primary}; font-weight:700; letter-spacing:0.2px; }}
    .hero .muted {{ color: {muted}; margin-top:2px; }}

    /* Sidebar card */
    .sidebar-card {{
        border-radius: 10px;
        padding: 10px 12px;
        background: linear-gradient(180deg, {panel}, {card});
        margin-bottom: 12px;
        border-left: 4px solid {primary};
        box-shadow: 0 6px 18px rgba(2,8,23,0.6);
    }}
    .sidebar-card h3 {{ margin: 0 0 4px 0; color: {text_color}; font-size:1.05rem; }}
    .sidebar-card .muted {{ color: {muted}; font-size: 0.92rem; }}

    /* KPI metric cards: polish */
    .stMetric {{
        border-radius: 10px;
        padding: 10px 14px;
        background: linear-gradient(180deg, rgba(11,95,255,0.04), rgba(11,95,255,0.02));
        border: 1px solid rgba(255,255,255,0.03);
    }}

    /* Quick-action / gallery cards */
    .quick-card {{
        border-radius: 10px;
        padding: 12px;
        background: linear-gradient(180deg, {panel}, {card});
        border: 1px solid rgba(255,255,255,0.03);
        box-shadow: 0 8px 20px rgba(2,8,23,0.6);
        margin-bottom: 10px;
    }}

    /* Gallery grid */
    .gallery-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 10px;
    }}
    .gallery-item img {{
        width: 100%;
        border-radius: 8px;
        box-shadow: 0 6px 18px rgba(2,8,23,0.6);
        border: 1px solid rgba(255,255,255,0.02);
    }}

    /* Buttons: subtle color and focus */
    .stButton>button {{
        background-color: {primary} !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        box-shadow: none !important;
        border: none !important;
    }}
    .stButton>button:hover {{ filter: brightness(1.08); }}
    .stButton>button:focus {{ outline: none; box-shadow: 0 0 0 4px rgba(11,95,255,0.12); }}

    /* Plotly container background to blend with theme */
    .stPlotlyChart>div, .stPlotlyChart>div>div {{
        background: transparent !important;
    }}

    /* Small utilities */
    .muted {{ color: {muted}; font-size: 0.95rem; }}

    /* Ensure Streamlit file_uploader browse button matches primary theme color */
    /* Targets the file uploader container and its internal button */
    div[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploader"] .stButton>button {{
        background: linear-gradient(90deg, {primary}, #0073E6) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        border: none !important;
        cursor: pointer !important;
        box-shadow: none !important;
    }}

    /* Reinforce Run Predictions button style (apply to all streamlit buttons for consistency) */
    .stButton>button, .stButton>button[role="button"] {{
        background: linear-gradient(90deg, {primary}, #0073E6) !important;
        color: #fff !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        font-weight: 700;
        cursor: pointer !important;
        border: none !important;
    }}
    .stButton>button:hover, .stButton>button[role="button"]:hover {{ filter: brightness(1.06); }}

    /* existing custom styles continue... */
    </style>
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
model_path_input = st.sidebar.text_input(
    "Model weights path",
    value=os.path.join(os.path.dirname(__file__), 'runs/detect/best_train_results2/weights/best.pt')
)
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
# Model Testing (Enhanced Layout only — logic preserved)
# ----------------------------
elif page == "Model Testing":
    st.title("YOLO Object Detection")
    st.markdown("Upload images (drag & drop or browse), tune thresholds, and run batch predictions. Results show annotated image and a detections summary.")

    # Page-specific small CSS for boxes, upload area, cards
    st.markdown(
        """
        <style>
        .param-box {
            border-radius:10px;
            padding:12px;
            background: linear-gradient(180deg, rgba(11,95,255,0.04), rgba(11,95,255,0.02));
            border:1px solid rgba(255,255,255,0.03);
            text-align:center;
            height:110px;
        }
        .param-label { font-size:0.95rem; color: rgba(230,240,255,0.9); margin-bottom:8px; }
        .param-value { font-weight:700; font-size:1.1rem; color: #E6F0FF; }

        .drop-area {
            border:2px dashed rgba(255,255,255,0.08);
            border-radius:8px;
            padding:28px;
            text-align:center;
            color: rgba(230,240,255,0.7);
            background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.02));
            min-height:120px;
        }

        .run-btn {
            display:block;
            width:100%;
            padding:12px 18px;
            background: linear-gradient(90deg, #0B5FFF, #0073E6);
            color:white;
            border-radius:10px;
            border:none;
            font-weight:700;
            font-size:1rem;
        }

        .img-card {
            border-radius:10px;
            padding:8px;
            background: linear-gradient(180deg, #0F2433, #0b1f2a);
            border:1px solid rgba(255,255,255,0.03);
        }

        .summary-card {
            border-radius:10px;
            padding:12px;
            background: linear-gradient(180deg, #0F2433, #071827);
            border:1px solid rgba(255,255,255,0.03);
        }

        .small-caption { color: rgba(230,240,255,0.7); font-size:0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top row: parameter boxes
    pcol1, pcol2, pcol3 = st.columns([1,1,1], gap="large")
    with pcol1:
        st.markdown('<div class="param-box"><div class="param-label">Confidence Threshold</div>', unsafe_allow_html=True)
        conf = st.slider("", 0.0, 1.0, 0.75 if 'conf' not in st.session_state else st.session_state.conf, 0.01, key="conf_slider")
        st.markdown(f'<div class="param-value">{conf:.2f}</div></div>', unsafe_allow_html=True)
    with pcol2:
        st.markdown('<div class="param-box"><div class="param-label">NMS IoU Threshold</div>', unsafe_allow_html=True)
        iou = st.slider("", 0.0, 1.0, 0.40 if 'iou' not in st.session_state else st.session_state.iou, 0.01, key="iou_slider")
        st.markdown(f'<div class="param-value">{iou:.2f}</div></div>', unsafe_allow_html=True)
    with pcol3:
        st.markdown('<div class="param-box"><div class="param-label">Inference Size</div>', unsafe_allow_html=True)
        imgsz = st.selectbox("", options=[320, 416, 640, 1024], index=2, key="imgsz_select")
        st.markdown(f'<div class="param-value">{imgsz}</div></div>', unsafe_allow_html=True)

    st.markdown("")  # small spacer

    # Middle: use only the Streamlit uploader (remove the custom drop-area + fake browse button)
    up_col, browse_col = st.columns([3,1], gap="medium")
    with up_col:
        # single uploader that shows drag-and-drop + browse; label kept for clarity
        uploaded_files = st.file_uploader("📤 Drag and drop images here (JPG, JPEG, PNG) — or click Browse files", type=['jpg','jpeg','png'], accept_multiple_files=True, key="file_uploader")
    with browse_col:
        # keep a small help hint; remove the non-functional HTML browse button
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="small-caption" style="margin-top:8px">Use the uploader on the left to select files.</div>', unsafe_allow_html=True)

    st.markdown("")  # spacer

    # Large Run Predictions button (centered)
    run_col1, run_col2, run_col3 = st.columns([1,2,1])
    with run_col2:
        run_btn = st.button("Run Predictions", key="run_predictions", help="Run inference on uploaded images")

    # Prediction logic (preserve original)
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

                    # Display results in polished two-column layout
                    res_col_left, res_col_right = st.columns([2,1], gap="large")
                    with res_col_left:
                        st.markdown(f'<div class="img-card">', unsafe_allow_html=True)
                        st.image(annotated, use_container_width=True)
                        st.markdown(f'<div class="small-caption" style="margin-top:6px">Saved to: <code>{out_path}</code></div>', unsafe_allow_html=True)
                        st.markdown(f'</div>', unsafe_allow_html=True)
                    with res_col_right:
                        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
                        st.markdown("<h4 style='margin:0 0 6px 0'>Detections Summary</h4>", unsafe_allow_html=True)
                        if detections:
                            # Build DataFrame with X,Y,W,H
                            rows = []
                            for d in detections:
                                x_min = d.get("xmin", 0)
                                y_min = d.get("ymin", 0)
                                x_max = d.get("xmax", 0)
                                y_max = d.get("ymax", 0)
                                w = x_max - x_min
                                h = y_max - y_min
                                rows.append({
                                    "Class": d.get("class", ""),
                                    "Confidence": round(d.get("confidence", 0), 2),
                                    "X": int(x_min),
                                    "Y": int(y_min),
                                    "W": int(w),
                                    "H": int(h)
                                })
                            df_res = pd.DataFrame(rows)
                            st.dataframe(df_res.style.format({"Confidence":"{:.2f}"}), use_container_width=True)
                            # Save csv to outputs/ (no download button shown)
                            csv_name = f"{os.path.splitext(uf.name)[0]}_predictions.csv"
                            csv_path = os.path.join(OUTPUT_FOLDER, csv_name)
                            df_res.to_csv(csv_path, index=False)
                            st.markdown(f'<div class="small-caption">Saved predictions CSV to: <code>{csv_path}</code></div>', unsafe_allow_html=True)
                            all_dets.append(df_res)
                        else:
                            st.info("No detections for this image.")
                        st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Failed on {uf.name}: {e}")
                progress.progress(int(idx/total*100))
            progress.empty()
            st.success("Batch predictions completed.")
            # aggregated CSV if multiple
            if all_dets:
                agg = pd.concat(all_dets, ignore_index=True)
                agg_path = os.path.join(OUTPUT_FOLDER, "predictions_aggregated.csv")
                agg.to_csv(agg_path, index=False)
                st.markdown(f'<div class="small-caption">Saved aggregated predictions to: <code>{agg_path}</code></div>', unsafe_allow_html=True)

    elif uploaded_files and not run_btn:
        st.info("Set parameters and click 'Run Predictions' to process uploaded images.")

#----------------------------
# Live Video Feed (Dynamic Input)
# ----------------------------
elif page == "Live Video Feed":
    st.title("Live Video Feed (via Flask API)")

    st.markdown(
        """
        <h5>Real-time webcam detection via Flask API</h5>
        <img src="http://127.0.0.1:8000/live" width="700" />
        """,
        unsafe_allow_html=True)

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