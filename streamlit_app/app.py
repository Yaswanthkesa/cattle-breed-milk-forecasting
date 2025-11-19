# app.py
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import os
import joblib

from predict_helper import load_cnn_lstm_model, predict_one, MILK_FEATURES, sequence_from_csv, find_and_load_scaler

st.set_page_config(page_title="Cow Breed & Milk Trait Predictor", layout="wide")

st.markdown("# 🐄 Cow Breed & Milk Trait Predictor (Xception+LSTM)")
st.markdown("Upload an image and a sequence (CSV with rows of milk features) — the app will resample the sequence to the model's expected length and show predicted milk traits in actual units when possible.")

# ---------- Sidebar: model selection ----------
st.sidebar.markdown("## Model / Encoder")
model_path = st.sidebar.text_input("Model path (.h5 or saved model folder)", value=r"C:\Users\Yaswanth kesa\Desktop\Streamlit\Xception_LSTM.h5")
if st.sidebar.button("Load model"):
    try:
        model = load_cnn_lstm_model(model_path)
        st.sidebar.success("Model loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        model = None
else:
    # try to auto-load default path if exists
    model = None
    try:
        if model_path and os.path.exists(model_path):
            model = load_cnn_lstm_model(model_path)
            st.sidebar.success("Model loaded.")
    except Exception:
        model = None

# optional label encoder upload
le = None
if st.sidebar.checkbox("Upload label encoder (.joblib)"):
    le_file = st.sidebar.file_uploader("label_encoder.joblib", type=["joblib"])
    if le_file is not None:
        try:
            le = joblib.load(le_file)
            st.sidebar.success("Encoder loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to load encoder: {e}")
            le = None

st.sidebar.markdown("---")
st.sidebar.markdown("## Options")
top_k = st.sidebar.number_input("Top-K breeds to show", min_value=1, max_value=20, value=5)

# ---------- Main UI ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 1) Upload image")
    img_file = st.file_uploader("Image (jpg/png)", type=["jpg", "jpeg", "png"])
    st.markdown("### 2) Upload/paste sequence (rows of 6 numbers in order: {})".format(", ".join(MILK_FEATURES)))
    seq_file = st.file_uploader("CSV sequence (any number of rows, 6 columns)", type=["csv"])
    seq_text = st.text_area("OR paste sequence rows (one row per line, 6 comma-separated values)", height=220)

with col2:
    st.markdown("Preview")
    img_preview = st.empty()
    st.markdown("---")
    st.markdown("Model info")
    if model is not None:
        st.write(f"Model name: **{getattr(model, 'name', 'N/A')}**")
        # try to infer seq len
        try:
            inputs = model.inputs
            if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
                seq_shape = inputs[1].shape
                dims = [int(x) if x is not None else None for x in seq_shape]
                expected_seq_len = dims[1]
                st.write(f"Model expects sequence length: **{expected_seq_len}** (will be auto-resampled)")
            else:
                st.write("Unable to infer model sequence input.")
        except Exception:
            st.write("Unable to infer model sequence input.")
    else:
        st.info("No model loaded yet. Use sidebar to load.")

# ---------- Helper parsing ----------
def parse_seq_file(fobj):
    return sequence_from_csv(fobj)

def parse_seq_text(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    rows = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",") if p.strip()]
        if len(parts) < len(MILK_FEATURES):
            raise ValueError(f"Each row must have at least {len(MILK_FEATURES)} numeric values.")
        rows.append([float(x) for x in parts[:len(MILK_FEATURES)]])
    return np.array(rows, dtype=np.float32)

# ---------- Predict button ----------
if st.button("Predict"):
    if model is None:
        st.error("Load the model first from the sidebar.")
        st.stop()

    if img_file is None:
        st.error("Upload an image first.")
        st.stop()

    # get image BGR
    try:
        pil = Image.open(img_file).convert("RGB")
        arr = np.array(pil)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        # preview
        preview = pil.copy()
        preview.thumbnail((420,420))
        img_preview.image(preview)
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        st.stop()

    # parse sequence
    seq_arr = None
    try:
        if seq_file is not None:
            seq_arr = parse_seq_file(seq_file)
        elif seq_text and seq_text.strip():
            seq_arr = parse_seq_text(seq_text)
        else:
            st.error("Provide a sequence (CSV or paste).")
            st.stop()
    except Exception as e:
        st.error(f"Failed to parse sequence: {e}")
        st.stop()

    # validate number of features per row
    if seq_arr.ndim != 2 or seq_arr.shape[1] != len(MILK_FEATURES):
        st.error(f"Sequence must be a 2D array with {len(MILK_FEATURES)} features per row. Got shape {seq_arr.shape}")
        st.stop()

    # pass model_dir so helper can find scaler
    model_dir = os.path.dirname(model_path) if model_path else None

    with st.spinner("Running model..."):
        try:
            result = predict_one(model, image_bgr=bgr, seq_array=seq_arr, model_dir=model_dir, breed_list=(list(le.classes_) if le is not None else None))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # show top-k breed probs
    probs = np.array(result.get("breed_probs") or [])
    if probs.size > 0:
        idxs = np.argsort(probs)[::-1][:top_k]
        df = []
        for r, i in enumerate(idxs, start=1):
            name = None
            if le is not None:
                try:
                    name = le.inverse_transform([int(i)])[0]
                except Exception:
                    name = None
            if name is None:
                # fallback to provided breed_list inside result if any
                name = result.get("predicted_breed") if i == result.get("predicted_index") else f"Class_{i}"
            df.append({"rank": r, "index": int(i), "breed": name, "prob": float(round(probs[i], 4))})
        st.markdown("### Top predicted breeds")
        st.table(pd.DataFrame(df))

    # show milk outputs (actual if available)
    if result.get("milk_regression_actual") is not None:
        st.markdown("### Predicted Milk Trait Values (actual units)")
        df_reg = pd.DataFrame([result["milk_regression_actual"]], columns=MILK_FEATURES).round(3)
        st.table(df_reg)
    elif result.get("milk_regression_scaled") is not None:
        st.warning("No scaler found to inverse-transform outputs; showing scaled/regression outputs.")
        df_reg = pd.DataFrame([result["milk_regression_scaled"]], columns=MILK_FEATURES).round(3)
        st.table(df_reg)
    else:
        st.info("Model did not return milk regression outputs.")

    if result.get("used_scaler_path"):
        st.caption(f"Scaler used: {result.get('used_scaler_path')}")

    st.success("Done ✅")
