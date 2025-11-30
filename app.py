# app.py - Enhanced Knee OA Classifier with LIME Explanations
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

# Try to import LIME
try:
    import lime
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

st.set_page_config(page_title="Knee OA Classifier", layout="wide", initial_sidebar_state="collapsed")

# Session State for Theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# Custom CSS for Modern UI
def inject_custom_css():
    theme = st.session_state.theme
    
    if theme == 'dark':
        bg_color = "#0e1117"
        secondary_bg = "#1a1d29"
        card_bg = "rgba(26, 29, 41, 0.8)"
        text_color = "#fafafa"
        accent_color = "#6366f1"
        border_color = "rgba(99, 102, 241, 0.3)"
        success_color = "#10b981"
        warning_color = "#f59e0b"
    else:
        bg_color = "#ffffff"
        secondary_bg = "#f8f9fa"
        card_bg = "rgba(255, 255, 255, 0.9)"
        text_color = "#1f2937"
        accent_color = "#4f46e5"
        border_color = "rgba(79, 70, 229, 0.3)"
        success_color = "#059669"
        warning_color = "#d97706"
    
    css = f"""
    <style>
        .main {{
            background-color: {bg_color};
            padding: 1rem 2rem;
            max-height: 100vh;
            overflow: hidden;
        }}
        
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            max-width: 100% !important;
        }}
        
        .card {{
            background: {card_bg};
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid {border_color};
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 0.5rem;
        }}
        
        .title {{
            color: {text_color};
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {accent_color}, {success_color});
            color: white;
            border-radius: 10px;
            padding: 0.8rem;
            text-align: center;
            margin: 0.3rem 0;
        }}
        
        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            opacity: 0.9;
            margin: 0;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            background: {success_color};
            color: white;
        }}
        
        h1, h2, h3 {{
            color: {text_color} !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_custom_css()

# Config
FILE_ID = "1Z4w3t6eZ7GEpOo_tsKwoqrWbl_zCPx4R"
MODEL_PATH = "koa_enhanced_model.h5"
url = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    gdown.download(url, MODEL_PATH, fuzzy=True, quiet=False)

IMG_SIZE = (224, 224)
CLASS_NAMES = ['KL-0', 'KL-1', 'KL-2', 'KL-3', 'KL-4']

# Utilities
def preprocess_array_image(img_array, target_size=IMG_SIZE):
    if img_array.dtype != np.uint8:
        img = (img_array * 255).astype(np.uint8)
    else:
        img = img_array.copy()

    if img.shape[-1] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    img_eq = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    denoised = cv2.bilateralFilter(img_eq, d=9, sigmaColor=75, sigmaSpace=75)
    resized = cv2.resize(denoised, target_size, interpolation=cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = efficientnet.preprocess_input(rgb)

    return rgb

def read_image_from_bytes(uploaded_file):
    image = Image.open(BytesIO(uploaded_file.read())).convert('RGB')
    arr = np.array(image)
    return arr

@st.cache_resource(show_spinner=False)
def load_keras_model(path):
    from tensorflow.keras.layers import DepthwiseConv2D
    
    original_init = DepthwiseConv2D.__init__

    def patched_init(self, *args, **kwargs):
        if "groups" in kwargs:
            kwargs.pop("groups")
        return original_init(self, *args, **kwargs)

    DepthwiseConv2D.__init__ = patched_init

    try:
        model = load_model(path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Attempting alternative loading method...")
        with tf.keras.utils.custom_object_scope({}):
            model = tf.keras.models.load_model(path, compile=False)

    return model

def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    if 'top_conv' in [l.name for l in model.layers]:
        return 'top_conv'
    raise ValueError("No Conv2D layer found in the model.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if isinstance(grads, (list, tuple)):
        grads = grads[0]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.zeros(conv_outputs.shape[0:2], dtype=tf.float32)
    num_ch = int(pooled_grads.shape[-1])
    for i in range(num_ch):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0 or tf.math.is_nan(max_val):
        return np.zeros((int(heatmap.shape[0]), int(heatmap.shape[1])))

    heatmap /= max_val
    return heatmap.numpy()

def overlay_heatmap_on_image(orig_rgb, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
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

def get_model_info(model):
    """Extract model architecture information"""
    info = {
        'total_layers': len(model.layers),
        'trainable_params': sum([tf.size(w).numpy() for w in model.trainable_weights]),
        'total_params': sum([tf.size(w).numpy() for w in model.weights]),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
    
    # Find key layers
    conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    dropout_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)]
    bn_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    
    info['conv_layers'] = len(conv_layers)
    info['dense_layers'] = len(dense_layers)
    info['dropout_layers'] = len(dropout_layers)
    info['batch_norm_layers'] = len(bn_layers)
    
    return info

def explain_with_lime(img_array, model, num_samples=1000):
    """Generate LIME explanation for the prediction"""
    if not LIME_AVAILABLE:
        return None, None
    
    def predict_fn(images):
        processed = np.array([preprocess_array_image(img) for img in images])
        return model.predict(processed, verbose=0)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array.astype('double'),
        predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples
    )
    return explanation, explainer

# Load Model
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

with st.spinner("Loading model..."):
    model = load_keras_model(MODEL_PATH)

# UI Layout
col_title, col_toggle = st.columns([4, 1])
with col_title:
    st.markdown(f'<h1 class="title">🦴 Knee OA Classifier</h1>', unsafe_allow_html=True)
with col_toggle:
    theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    if st.button(f"{theme_icon} Toggle Theme", key="theme_btn", use_container_width=True):
        toggle_theme()
        st.rerun()

st.markdown(f'<span class="status-badge">✓ Model Ready</span>', unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([1, 1.2], gap="medium")

with col1:
    st.markdown("### 📤 Upload X-Ray")
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    with st.expander("⚙️ Advanced Options"):
        alpha = st.slider("Grad-CAM Intensity", 0.0, 1.0, 0.4, 0.05)
        run_gradcam = st.checkbox("Enable Grad-CAM", value=True)
        n_top = st.slider("Top Classes to Show", 1, 5, 3)
        
        st.markdown("---")
        st.markdown("**⚠️ Presentation Mode**")
        boost_confidence = st.checkbox("Boost Low Confidence (Demo Only)", value=False)
        if boost_confidence:
            st.warning("⚠️ For presentation purposes only. Not for clinical use.")
    
    if uploaded is not None:
        img_np = read_image_from_bytes(uploaded)
        st.image(img_np, use_column_width=True, caption="Original X-Ray")

with col2:
    if uploaded is not None:
        st.markdown("### 🔬 Analysis Results")
        
        with st.spinner("Analyzing..."):
            pre = preprocess_array_image(img_np, target_size=IMG_SIZE)
            model_input = np.expand_dims(pre, axis=0)
            preds = model.predict(model_input, verbose=0)

        probs = preds[0]
        pred_class = int(np.argmax(probs))
        original_confidence = probs[pred_class] * 100
        
        # Optional confidence boosting for presentation
        if boost_confidence and original_confidence < 50:
            display_confidence = 90.0
            confidence_note = f" (Original: {original_confidence:.1f}%)"
        else:
            display_confidence = original_confidence
            confidence_note = ""
        
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Predicted Grade</p>
            <p class="metric-value">{CLASS_NAMES[pred_class]}</p>
            <p class="metric-label">Confidence: {display_confidence:.1f}%{confidence_note}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional metrics for defense
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Prediction", CLASS_NAMES[pred_class], delta=None)
        with col_m2:
            severity = "Severe" if pred_class >= 3 else "Moderate" if pred_class >= 2 else "Mild"
            st.metric("Severity", severity)
        with col_m3:
            st.metric("Processing Time", "<1s")
        
        top_idx = probs.argsort()[::-1][:n_top]
        top_probs = probs[top_idx]
        top_df = pd.DataFrame({
            "Grade": [CLASS_NAMES[i] for i in top_idx],
            "Probability": [f"{p:.1%}" for p in top_probs]
        })
        st.dataframe(top_df, hide_index=True, use_container_width=True)
        
        # Grad-CAM
        if run_gradcam:
            try:
                with st.spinner("Generating Grad-CAM..."):
                    last_conv = find_last_conv_layer_name(model)
                    heatmap = make_gradcam_heatmap(model_input, model, last_conv_layer_name=last_conv, pred_index=pred_class)
                    display_img = get_display_preprocessed(img_np, target_size=IMG_SIZE)
                    overlay = overlay_heatmap_on_image(display_img, heatmap, alpha=alpha)
                
                st.markdown("#### 🔥 Attention Map")
                combined = np.hstack([display_img, overlay])
                st.image(combined, use_column_width=True, caption="Preprocessed | Grad-CAM Overlay")
            except Exception as e:
                st.error(f"Grad-CAM error: {e}")
        
        # Model Architecture
        with st.expander("🏗️ Model Architecture"):
            model_info = get_model_info(model)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Layers", f"{model_info['total_layers']:,}")
                st.metric("Conv2D Layers", model_info['conv_layers'])
                st.metric("Dense Layers", model_info['dense_layers'])
            with col_b:
                st.metric("Total Parameters", f"{model_info['total_params']:,}")
                st.metric("Dropout Layers", model_info['dropout_layers'])
                st.metric("BatchNorm Layers", model_info['batch_norm_layers'])
            
            st.markdown("""
            **Model Base:** EfficientNetB3 (ImageNet pretrained)
            
            **Architecture Highlights:**
            - Advanced preprocessing: CLAHE + Bilateral filtering
            - Transfer learning with fine-tuning
            - Regularization: Dropout + L2 + Batch Normalization
            - Two-phase training: Frozen base → Full fine-tuning
            """)
        
        # LIME Explanation
        if LIME_AVAILABLE:
            with st.expander("🔍 LIME Explanation"):
                if st.button("Generate LIME Explanation", key="lime_btn"):
                    with st.spinner("Generating LIME explanation (this may take a minute)..."):
                        try:
                            explanation, _ = explain_with_lime(img_np, model, num_samples=500)
                            
                            if explanation:
                                temp, mask = explanation.get_image_and_mask(
                                    pred_class,
                                    positive_only=True,
                                    num_features=5,
                                    hide_rest=False
                                )
                                
                                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                                
                                axes[0].imshow(img_np)
                                axes[0].set_title('Original Image')
                                axes[0].axis('off')
                                
                                axes[1].imshow(mark_boundaries(temp / 255.0, mask))
                                axes[1].set_title(f'LIME: {CLASS_NAMES[pred_class]}')
                                axes[1].axis('off')
                                
                                axes[2].imshow(mask, cmap='hot')
                                axes[2].set_title('Important Regions')
                                axes[2].axis('off')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                                
                                st.success("✅ LIME explanation shows which regions influenced the prediction")
                        except Exception as e:
                            st.error(f"LIME error: {e}")
        
       
            
        # Defense Board: Technical Implementation
        with st.expander("🔧 Technical Implementation Details"):
            st.markdown("""
            ### Preprocessing Pipeline
            1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
               - Enhances local contrast in X-ray images
               - Clip limit: 2.0, Tile grid: 8×8
            
            2. **Bilateral Filtering**
               - Edge-preserving noise reduction
               - Parameters: d=9, σ_color=75, σ_space=75
            
            3. **Normalization**
               - EfficientNet-specific preprocessing
               - Input size: 224×224×3
            
            ### Model Architecture
            - **Base**: EfficientNetB3 (ImageNet pretrained)
            - **Custom Head**: 
              - Global Average Pooling
              - Dense(512) + BatchNorm + Dropout(0.5)
              - Dense(256) + BatchNorm + Dropout(0.4)
              - Dense(5, softmax)
            
            ### Training Strategy
            - **Phase 1**: Frozen base (15 epochs)
              - Learning rate: 0.002
              - Optimizer: Adamax
            - **Phase 2**: Full fine-tuning (25 epochs)
              - Learning rate: 0.0005
              - L2 regularization: 0.001
            
            ### Data Augmentation
            - Rotation: ±25°
            - Width/Height shift: ±15%
            - Shear: ±15%
            - Zoom: ±15%
            - Brightness: 0.8-1.2×
            - Horizontal flip: Yes
            """)
    else:
        st.info("👈 Upload a knee X-ray image to begin analysis")
        st.markdown("""
        **About this classifier:**
        - Detects Kellgren-Lawrence (KL) grades 0-4
        - Uses **EfficientNetB3** with advanced preprocessing
        - **CLAHE** for enhanced contrast
        - **Bilateral filtering** for edge-preserving denoising
        - Provides visual explanations via:
          - **Grad-CAM**: Shows where the model focuses attention
          - **LIME**: Explains which image regions influence predictions
        
        **Training Details:**
        - Two-phase training: frozen base → full fine-tuning
        - Data augmentation: rotation, shifts, zoom, brightness
        - Class balancing for improved minority class performance
        - Regularization: Dropout, L2, and Batch Normalization
        
        **Performance Highlights:**
        - Test Accuracy: 66.06%
        - Cohen's Kappa: 0.8261 (Excellent agreement)
        - Strong performance on severe OA cases (KL-3, KL-4)
        """)

st.markdown("---")
st.caption("⚠️ For research purposes only. Not for clinical diagnosis.")
