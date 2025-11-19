# predict_helper.py
"""
Robust helper utilities for Xception+LSTM Streamlit app.

Features:
- load_cnn_lstm_model(model_path)
- find_and_load_scaler(search_paths)
- enhance_and_prepare(img_bgr, target_size, use_xception_preprocess)
- sequence_from_csv(path_or_buffer)
- prepare_sequence(seq_array, scaler=None, target_len=None, feature_count=None)
- predict_one(model, image_bgr=None, image_path=None, seq_array=None, scaler=None, breed_list=None, model_dir=None)
"""

import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import tensorflow as tf

# ---------------- CONFIG ----------------
MILK_FEATURES = ["Milk_Yield_L", "Fat_pct", "Protein_pct", "Lactose_pct", "SNF_pct", "Total_Solids_pct"]
DEFAULT_IMG_SIZE = (299, 299)  # (width, height) default Xception
DEFAULT_SEQ_LEN = 28  # fallback if model inference fails

SCALER_CANDIDATES = [
    "milk_scaler.joblib", "scaler.joblib", "minmax_scaler.joblib", "milk_scaler.pkl",
    os.path.join("models", "milk_scaler.joblib")
]

# ---------------- Model loading ----------------
def load_cnn_lstm_model(model_path, custom_objects=None):
    """
    Load a Keras model from disk. Provide custom_objects if your model used custom layers/losses.
    """
    if custom_objects is None:
        custom_objects = {'mse': MeanSquaredError()}
    model = load_model(model_path, custom_objects=custom_objects)
    return model

# ---------------- Scaler discovery ----------------
def find_and_load_scaler(search_paths=None):
    """
    Search for a scaler file in provided paths (dirs) and current working dir.
    Returns (scaler_obj, path) or (None, None).
    """
    candidates = SCALER_CANDIDATES.copy()
    # expand with search_paths if provided
    if search_paths:
        expanded = []
        for p in search_paths:
            for c in SCALER_CANDIDATES:
                expanded.append(os.path.join(p, c))
        candidates = expanded + candidates

    # ensure cwd candidate also present
    if os.getcwd() not in (search_paths or []):
        for c in SCALER_CANDIDATES:
            candidates.append(os.path.join(os.getcwd(), c))

    for fn in candidates:
        if fn and os.path.exists(fn):
            try:
                sc = joblib.load(fn)
                return sc, fn
            except Exception:
                continue
    return None, None

# ---------------- Image preprocessing ----------------
def _xception_preprocess(img_rgb):
    """Use tf.keras.applications.xception.preprocess_input on RGB image array."""
    return tf.keras.applications.xception.preprocess_input(img_rgb.astype(np.float32))

def _simple_normalize(img_rgb):
    """Scale 0-255 to 0-1 float32."""
    return (img_rgb.astype("float32") / 255.0)

def enhance_and_prepare(img_bgr, target_size=DEFAULT_IMG_SIZE, use_xception_preprocess=False):
    """
    CLAHE enhancement -> resize -> convert to RGB -> preprocess (xception or simple).
    Input: BGR uint8
    Returns: HxWx3 preprocessed RGB array (not batched).
    """
    # cv2.resize expects (width, height) tuple
    img = cv2.resize(img_bgr, target_size)
    # CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    if use_xception_preprocess:
        return _xception_preprocess(final_rgb)
    return _simple_normalize(final_rgb)

# ---------------- Sequence utilities ----------------
def resample_sequence(seq, target_len):
    """
    Resample 2D sequence seq (T, F) to (target_len, F) using linear interpolation.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError("resample_sequence expects 2D array (T, F).")
    T, F = seq.shape
    if T == target_len:
        return seq.copy()
    orig_idx = np.linspace(0.0, 1.0, T)
    target_idx = np.linspace(0.0, 1.0, target_len)
    new_seq = np.zeros((target_len, F), dtype=np.float32)
    for f in range(F):
        new_seq[:, f] = np.interp(target_idx, orig_idx, seq[:, f])
    return new_seq

def prepare_sequence(seq_array, scaler=None, target_len=None, feature_count=None):
    """
    Prepare sequence to shape (1, target_len, F), scaling with provided scaler or fallback MinMaxScaler.
    - seq_array: numpy array shape (T, F)
    - scaler: fitted sklearn scaler or None
    - target_len: int or None; if provided and different from T, sequence will be resampled
    - feature_count: expected feature count F (optional)
    Returns (seq_scaled (1,target_len,F), scaler_used)
    """
    seq = np.asarray(seq_array, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"seq_array must be 2D (T,F). Got {seq.shape}")
    T, F = seq.shape
    if feature_count is not None and F != feature_count:
        raise ValueError(f"Expected {feature_count} features per timestep, got {F}")

    if target_len is None:
        target_len = T

    if T != target_len:
        seq = resample_sequence(seq, target_len)
        T = target_len

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(seq)

    seq_scaled = scaler.transform(seq).reshape(1, T, F)
    return seq_scaled, scaler

def sequence_from_csv(path_or_buffer):
    """
    Read CSV and return numpy array shape (T, F). Tries MILK_FEATURES column names first,
    otherwise picks first numeric columns equal to len(MILK_FEATURES).
    """
    df = pd.read_csv(path_or_buffer)
    cols = [c for c in df.columns if c in MILK_FEATURES]
    if len(cols) == len(MILK_FEATURES):
        arr = df[cols].values.astype(np.float32)
    else:
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numcols) < len(MILK_FEATURES):
            arr = df[numcols].values.astype(np.float32)
        else:
            arr = df[numcols[:len(MILK_FEATURES)]].values.astype(np.float32)
    return arr

# ---------------- Model introspection helpers ----------------
def _infer_image_input_shape_and_xception_flag(model):
    """
    Try to infer (width,height,channels) and whether to use xception preprocess.
    Returns (width, height, channels, use_xception_pre)
    """
    # defaults
    w, h, c = DEFAULT_IMG_SIZE[0], DEFAULT_IMG_SIZE[1], 3
    use_xception_pre = False
    try:
        inputs = model.inputs
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 1:
            img_shape = inputs[0].shape
        else:
            img_shape = model.input.shape
        dims = [int(x) if x is not None else None for x in img_shape]
        if len(dims) >= 4:
            _, h_dim, w_dim, c_dim = dims[:4]
            if h_dim and w_dim:
                h, w = h_dim, w_dim
            if c_dim:
                c = c_dim
            if (h == 299 and w == 299) or ('xception' in (getattr(model, "name", "") or "").lower()):
                use_xception_pre = True
        elif len(dims) == 3:
            h_dim, w_dim, c_dim = dims
            if h_dim and w_dim:
                h, w = h_dim, w_dim
            if c_dim:
                c = c_dim
            if (h == 299 and w == 299) or ('xception' in (getattr(model, "name", "") or "").lower()):
                use_xception_pre = True
    except Exception:
        pass
    return (w, h, c, use_xception_pre)

def _infer_expected_seq_len_and_feat_from_model(model):
    """
    Try to infer expected sequence length (int) and feature count (int) from second input shape.
    Returns (seq_len or None, feat_count or None).
    """
    try:
        inputs = model.inputs
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            seq_shape = inputs[1].shape
            dims = [int(x) if x is not None else None for x in seq_shape]
            if len(dims) >= 3:
                seq_len = dims[1]
                feat = dims[2]
                return (int(seq_len) if seq_len is not None else None, int(feat) if feat is not None else None)
    except Exception:
        pass
    return (None, None)

# ---------------- Prediction wrapper ----------------
def predict_one(model, *, image_bgr=None, image_path=None, seq_array=None, scaler=None, breed_list=None, model_dir=None):
    """
    Run a single prediction.
    - model: loaded Keras model expecting [image_batch, seq_batch]
    - image_bgr or image_path required
    - seq_array: numpy array (T, F)
    - scaler: fitted scaler (optional). If None, helper will try to find saved scaler in model_dir or cwd.
    - breed_list: optional list mapping indices -> class names
    - model_dir: directory to search for scaler candidates
    Returns dict with:
      - breed_probs
      - predicted_index
      - predicted_breed
      - milk_regression_scaled (raw model output, if present)
      - milk_regression_actual (inverse-transformed, if scaler available)
      - used_scaler_path (path or None)
    """
    if image_bgr is None and image_path is None:
        raise ValueError("Provide image_bgr or image_path")
    if seq_array is None:
        raise ValueError("Provide seq_array (T, F)")

    # read image if only path provided
    if image_bgr is None:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

    # infer image input expectations
    inferred_w, inferred_h, inferred_c, use_xception_pre = _infer_image_input_shape_and_xception_flag(model)
    inferred_img_size = (inferred_w, inferred_h)

    # preprocess image
    img_pre = enhance_and_prepare(image_bgr, target_size=inferred_img_size, use_xception_preprocess=use_xception_pre)
    img_input = np.expand_dims(img_pre, axis=0)  # batch

    # scaler handling: if not provided, try to find one
    used_scaler = scaler
    used_scaler_path = None
    if used_scaler is None:
        search_paths = []
        if model_dir:
            search_paths.append(model_dir)
        search_paths.append(os.getcwd())
        sc, scpath = find_and_load_scaler(search_paths)
        if sc is not None:
            used_scaler = sc
            used_scaler_path = scpath

    # infer expected seq length and feature count from model
    expected_seq_len, expected_feat_count = _infer_expected_seq_len_and_feat_from_model(model)
    if expected_feat_count is None:
        expected_feat_count = len(MILK_FEATURES)

    # prepare (resample + scale) sequence
    seq_scaled, used_scaler = prepare_sequence(seq_array, scaler=used_scaler, target_len=expected_seq_len, feature_count=expected_feat_count)

    # perform prediction
    preds = model.predict([img_input, seq_scaled], verbose=0)

    # handle outputs: common pattern [breed_probs, milk_regression]
    milk_scaled = None
    milk_actual = None
    if isinstance(preds, (list, tuple)) and len(preds) == 2:
        breed_probs, milk_reg = preds
        milk_scaled = np.asarray(milk_reg).ravel().tolist()
        # attempt inverse-transform if scaler available
        if used_scaler is not None:
            try:
                inv = used_scaler.inverse_transform(np.asarray(milk_scaled).reshape(1, -1))[0].tolist()
                milk_actual = inv
            except Exception:
                milk_actual = None
    else:
        breed_probs = preds
        milk_scaled = None
        milk_actual = None

    breed_probs = np.asarray(breed_probs).ravel()
    pred_idx = int(np.argmax(breed_probs)) if breed_probs.size > 0 else None
    pred_breed = None
    if breed_list is not None and pred_idx is not None and 0 <= pred_idx < len(breed_list):
        pred_breed = breed_list[pred_idx]

    return {
        "breed_probs": breed_probs.tolist(),
        "predicted_index": pred_idx,
        "predicted_breed": pred_breed,
        "milk_regression_scaled": milk_scaled,
        "milk_regression_actual": milk_actual,
        "used_scaler_path": used_scaler_path
    }
