# app.py
import os
from io import BytesIO
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import efficientnet
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import onnxruntime as ort
import gdown

st.set_page_config(page_title="Knee OA (KL) Classifier", layout="centered")

# -----------------------
# Config - edit if needed
# -----------------------
import gdown

FILE_ID = "1Z4w3t6eZ7GEpOo_tsKwoqrWbl_zCPx4R"
MODEL_PATH = "koa_enhanced_model.h5"
url = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    gdown.download(url, MODEL_PATH, fuzzy=True, quiet=False)

IMG_SIZE = (224, 224)          # model input size you used in training
CLASS_NAMES = ['KL-0', 'KL-1', 'KL-2', 'KL-3', 'KL-4']

# -----------------------
# Utilities: preprocessing
# -----------------------
def preprocess_array_image(img_array, target_size=IMG_SIZE):
    """
    Preprocessing pipeline matching your training:
      - Accepts a numpy array in RGB or BGR format (PIL -> RGB np array).
      - Convert to BGR for OpenCV ops consistently.
      - CLAHE on grayscale, convert back, bilateral denoise, resize (LANCZOS4),
      - Convert to RGB then efficientnet.preprocess_input
    Returns preprocessed array shaped (H,W,3) float32 ready for model (NOT batched).
    """
    # If image is float in [0,1] or uint8
    if img_array.dtype != np.uint8:
        img = (img_array * 255).astype(np.uint8)
    else:
        img = img_array.copy()

    # Ensure RGB -> convert to BGR for OpenCV ops
    if img.shape[-1] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        # if single channel, convert to BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert to grayscale for CLAHE
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    img_eq = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    # Bilateral denoising (preserve edges)
    denoised = cv2.bilateralFilter(img_eq, d=9, sigmaColor=75, sigmaSpace=75)

    # Resize to target_size (LANCZOS4 similar to your training)
    resized = cv2.resize(denoised, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Convert back to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

    # EfficientNet preprocessing (this includes scaling/normalization expected by the model)
    rgb = efficientnet.preprocess_input(rgb)

    return rgb

def read_image_from_bytes(uploaded_file):
    """Return numpy RGB image from uploaded bytes (PIL -> RGB np array)."""
    image = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
    arr = np.array(image)
    return arr

# -----------------------
# Model loading (cached)
# -----------------------
# -----------------------
# Model loading (cached)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_keras_model(path):

    # --- PATCH: Drop "groups" arg which old TF can't handle ---
    from tensorflow.keras.layers import DepthwiseConv2D
    import inspect
    
    original_init = DepthwiseConv2D.__init__

    def patched_init(self, *args, **kwargs):
        # Remove unsupported "groups" if present
        if "groups" in kwargs:
            kwargs.pop("groups")
        return original_init(self, *args, **kwargs)

    DepthwiseConv2D.__init__ = patched_init
    # ----------------------------------------------------------

    try:
        model = load_model(path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Attempting alternative loading method...")

        with tf.keras.utils.custom_object_scope({}):
            model = tf.keras.models.load_model(path, compile=False)

    return model


# Load model
model_session = load_keras_model(MODEL_PATH)
st.success("h5 Model loaded ✅")
# -----------------------
# Grad-CAM utilities
# -----------------------
def find_last_conv_layer_name(model):
    # Search backwards for Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # fallback common EfficientNet conv layer
    if 'top_conv' in [l.name for l in model.layers]:
        return 'top_conv'
    raise ValueError("No Conv2D layer found in the model. Provide layer name manually.")
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    img_array: preprocessed image batch (1,H,W,3)
    returns heatmap HxW normalized 0-1 (numpy)
    Works for single-output or multi-output models and avoids list/tuple indexing errors.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)

    # Build grad model safely using model.input (not [model.inputs])
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Run forward pass and watch for returned types (lists or tensors)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # If conv_outputs returned as list, take first element
        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]

        # If predictions returned as list (multi-output), try to use the first output
        if isinstance(predictions, (list, tuple)):
            # if model has multiple outputs and the final classification is a single vector,
            # we assume predictions[0] is the class logits/probs. Adjust if your model differs.
            predictions = predictions[0]

        # determine predicted class index
        if pred_index is None:
            # predictions shape expected (1, num_classes)
            pred_index = tf.argmax(predictions[0])
        # class_channel will be shape (batch,) (we want to differentiate that scalar)
        class_channel = predictions[:, pred_index]

    # Compute gradients of the target class w.r.t. the last conv outputs
    grads = tape.gradient(class_channel, conv_outputs)
    # If grads is list, grab first item
    if isinstance(grads, (list, tuple)):
        grads = grads[0]

    # pooled_grads: mean over height and width
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # conv_outputs should be a tensor (H, W, channels) after removing batch dim
    conv_outputs = conv_outputs[0]

    # Compute weighted sum of feature maps
    # ensure pooled_grads is broadcastable
    heatmap = tf.zeros(conv_outputs.shape[0:2], dtype=tf.float32)
    # iterate channels safely using python int(...) for shape elements
    num_ch = int(pooled_grads.shape[-1])
    for i in range(num_ch):
        # pooled_grads[i] is a scalar tensor
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    # avoid division by zero
    if max_val == 0 or tf.math.is_nan(max_val):
        return np.zeros((int(heatmap.shape[0]), int(heatmap.shape[1])))

    heatmap /= max_val
    return heatmap.numpy()

def overlay_heatmap_on_image(orig_rgb, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    orig_rgb: original RGB image (H,W,3) in uint8 or float
    heatmap: HxW in 0-1
    Returns overlay image (H, W, 3) uint8
    """
    if orig_rgb.dtype != np.uint8:
        img = np.clip(orig_rgb * 255, 0, 255).astype(np.uint8)
    else:
        img = orig_rgb.copy()

    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# -----------------------
# UI
# -----------------------
st.title("Knee Osteoarthritis (KL) Grade Classifier")
st.write("Upload a knee X-ray image (jpg/png). The app will preprocess using CLAHE + denoise and run your EfficientNetB3 model.")

# Model loader / status
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Put your `best_model.h5` next to this app.")
    st.stop()

with st.spinner("Loading model..."):
    model = load_keras_model(MODEL_PATH)
st.success("Model loaded ✅")
st.write(f"Model: `{MODEL_PATH}` — input size: {IMG_SIZE[0]}×{IMG_SIZE[1]}")

# Sidebar controls
st.sidebar.header("Options")
alpha = st.sidebar.slider("Grad-CAM alpha (overlay intensity)", 0.0, 1.0, 0.4, 0.05)
run_gradcam = st.sidebar.checkbox("Compute Grad-CAM overlay", value=True)
n_top = st.sidebar.slider("Show top-N classes", 1, 5, 3)

uploaded = st.file_uploader("Upload knee X-ray (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    # Read uploaded image
    img_np = read_image_from_bytes(uploaded)
    # Show original
    st.subheader("Uploaded image")
    st.image(img_np, use_container_width=True)


    # Let user optionally crop/rotate? (skip for simplicity) - we just resize as user said
    st.write("Preprocessing: CLAHE → bilateral denoise → resize → EfficientNet preprocess")

    with st.spinner("Preprocessing image..."):
        pre = preprocess_array_image(img_np, target_size=IMG_SIZE)   # H,W,3 float32 in ENet format
        model_input = np.expand_dims(pre, axis=0)  # (1,H,W,3)

    with st.spinner("Running prediction..."):
       preds = model.predict(model_input)


    probs = preds[0]
    top_idx = probs.argsort()[::-1][:n_top]
    top_probs = probs[top_idx]

    # Display results
    st.subheader("Prediction")
    pred_class = int(np.argmax(probs))
    st.markdown(f"**Predicted KL grade:** **{CLASS_NAMES[pred_class]}**")
    st.markdown(f"**Confidence:** {probs[pred_class]*100:.2f}%")

    # Show top N probabilities table
    top_df = pd.DataFrame({
        "class": [CLASS_NAMES[i] for i in top_idx],
        "probability": top_probs
    })
    st.table(top_df.style.format({"probability": "{:.3f}"}))

    # Bar chart of all classes
    probs_df = pd.DataFrame({"class": CLASS_NAMES, "prob": probs})
    st.bar_chart(data=probs_df.set_index('class'))

    # Grad-CAM
    if run_gradcam:
        try:
            with st.spinner("Computing Grad-CAM..."):
                last_conv = find_last_conv_layer_name(model)
                heatmap = make_gradcam_heatmap(model_input, model, last_conv_layer_name=last_conv, pred_index=pred_class)
                # For overlay, use the original preprocessed RGB *before* efficientnet preprocessing.
                # Our preprocess_array_image applies efficientnet.preprocess_input to the RGB array.
                # To get the original RGB after CLAHE/denoise/resize, recompute without EN preprocess:
                # We'll reuse preprocess_array_image but reverse EN preprocessing by undoing mean scale is complicated.
                # Simpler: redo the CLAHE/denoise/resize on the original RGB and use that for overlay.
                # Convert uploaded image to BGR->CLAHE->denoise->resize -> RGB (uint8)
                def get_display_preprocessed(orig_np, target_size=IMG_SIZE):
                    if orig_np.dtype != np.uint8:
                        tmp = (orig_np * 255).astype(np.uint8)
                    else:
                        tmp = orig_np.copy()
                    bgr = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    eq = clahe.apply(gray)
                    eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
                    den = cv2.bilateralFilter(eq_bgr, d=9, sigmaColor=75, sigmaSpace=75)
                    resized = cv2.resize(den, target_size, interpolation=cv2.INTER_LANCZOS4)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    return rgb
                display_img = get_display_preprocessed(img_np, target_size=IMG_SIZE)
                overlay = overlay_heatmap_on_image(display_img, heatmap, alpha=alpha)
                st.subheader("Grad-CAM Overlay")
               # Single side-by-side image
                st.image(np.hstack([display_img, overlay]), use_container_width=True, caption="Preprocessed | Overlay")


        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")

    # Option to download prediction results / overlay
    st.markdown("---")
    st.caption("Model & preprocessing must match the training pipeline for reliable results.")
else:
    st.info("Upload an X-ray to run prediction and Grad-CAM.")
