# app.py
import json
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from tensorflow.keras.models import load_model

MODEL_PATH = "trained_plant_disease_model.keras"
CLASS_MAP_PATH = "class_indices.json"
EXPECTED_NUM_CLASSES = 38  # set to None to skip this check

st.title("🌿 Plant Disease Classifier (Stabilized)")
st.caption("Center-crop + resize; averaged over 3 common preprocessing modes to avoid skew.")

@st.cache_resource
def load_cls_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_class_names():
    with open(CLASS_MAP_PATH, "r") as f:
        raw = json.load(f)  # can be {"0":"name"} or {"name":0}
    # Support both formats
    try:
        # {"0":"name", ...}
        mapping = {int(k): v for k, v in raw.items()}
        names = [mapping[i] for i in range(max(mapping.keys()) + 1)]
    except Exception:
        # {"name":0, ...}
        names = [None] * (max(raw.values()) + 1)
        for name, idx in raw.items():
            names[idx] = name
        if any(n is None for n in names):
            raise ValueError("class_indices.json must have contiguous indices from 0.")
    return names

def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def preprocess_variants(pil_img, W, H):
    """Return list of (name, array[N,H,W,C]) for the 3 common preprocessing modes."""
    img = pil_img.convert("RGB")
    img = center_crop_to_square(img).resize((W, H))
    rgb = np.array(img, dtype=np.float32)

    # A) RGB scaled to [0,1]
    a = (rgb / 255.0)[None, ...]

    # B) TF mode [-1, 1]
    b = ((rgb / 127.5) - 1.0)[None, ...]

    # C) Caffe BGR mean subtraction
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    bgr = rgb[:, :, ::-1]          # RGB -> BGR
    c = (bgr - mean)[None, ...]

    return [("rgb_01", a), ("rgb_tf", b), ("bgr_caffe", c)]

def to_probs(logits_like):
    """Turn logits or already-softmaxed output into a probability vector."""
    vec = logits_like[0] if logits_like.ndim == 2 else logits_like
    vec = np.asarray(vec, dtype=np.float32)
    s = vec.sum()
    if not np.isfinite(s) or abs(s - 1.0) > 1e-3:
        e = np.exp(vec - np.max(vec))
        vec = e / e.sum()
    return vec

def detect_channel_order(in_shape):
    # (None,H,W,C) or (None,C,H,W)
    if isinstance(in_shape, (list, tuple)) and len(in_shape) == 4:
        if in_shape[-1] == 3:
            return "channels_last"
        if in_shape[1] == 3:
            return "channels_first"
    return "channels_last"

def match_model_input(arr_NHWC, order):
    if order == "channels_last":
        return arr_NHWC
    return np.transpose(arr_NHWC, (0, 3, 1, 2)).astype(np.float32)  # (N,C,H,W)

# ---- load model/labels
try:
    model = load_cls_model()
    class_names = load_class_names()
except Exception as e:
    st.error("Failed to load model or labels.")
    st.exception(e)
    st.stop()

# shapes & classes
in_shape = model.input_shape
out_shape = model.output_shape
order = detect_channel_order(in_shape)

# Determine H, W safely
try:
    if order == "channels_last":
        H, W = in_shape[1], in_shape[2]
    else:
        H, W = in_shape[2], in_shape[3]
    if H is None or W is None:
        H, W = 224, 224
except Exception:
    H, W = 224, 224

# Num classes from model output
try:
    # Usually (None, num_classes)
    num_model_classes = int(out_shape[-1])
except Exception:
    st.error("Could not determine number of output classes from model.output_shape.")
    st.write(f"output_shape: {out_shape}")
    st.stop()

if EXPECTED_NUM_CLASSES is not None and num_model_classes != EXPECTED_NUM_CLASSES:
    st.error(f"Model has {num_model_classes} classes; expected {EXPECTED_NUM_CLASSES}.")
    st.stop()

if num_model_classes != len(class_names):
    st.error("Label mismatch: model output units != number of labels in JSON.")
    st.stop()

# ---- UI
uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        # Auto-rotate (this alone fixes many “no output” cases)
        img = ImageOps.exif_transpose(Image.open(uploaded))

        # Build 3 variants in channels-last, then align to model
        variants = preprocess_variants(img, W, H)
        probs_list = []
        for _, arr_CL in variants:
            arr = match_model_input(arr_CL, order)
            pred = model.predict(arr, verbose=0)
            probs_list.append(to_probs(pred))

        probs_avg = np.mean(probs_list, axis=0)

        top1 = int(np.argmax(probs_avg))
        top5 = np.argsort(-probs_avg)[:5]

        st.subheader("Predicted Disease")
        st.write(class_names[top1])

        with st.expander("Top-5 (useful to confirm all classes work)"):
            for i in top5:
                st.write(f"- {class_names[i]}: {probs_avg[i]*100:.2f}%")

    except Exception as e:
        # If “nothing shows”, you’ll now see the actual error here.
        st.error("Error while processing or predicting. See details below:")
        st.exception(e)
