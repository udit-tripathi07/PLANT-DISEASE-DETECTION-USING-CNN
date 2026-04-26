import os
os.environ["KERAS_BACKEND"] = "tensorflow"   # must be FIRST before any keras/tf import

import streamlit as st
import numpy as np
from PIL import Image
import time

# ---------- IMPORT KERAS CORRECTLY ----------
# Standalone keras 3.x must be used — NOT tf.keras (Keras 2)
try:
    import keras
    _KERAS_VERSION = keras.__version__
except ImportError:
    import tensorflow as tf
    keras = tf.keras
    _KERAS_VERSION = tf.__version__

# TF still needed for img_to_array
import tensorflow as tf

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
* { box-sizing: border-box; }
html, body, .stApp { background-color: #0a0f0a; color: #e8f5e8; font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1f0d 0%, #0a150a 100%); border-right: 1px solid #1a3a1a; }
section[data-testid="stSidebar"] * { color: #c8e6c8 !important; }
.hero-banner { background: linear-gradient(135deg, #0d2b0d 0%, #1a4a1a 40%, #0f3320 100%); border: 1px solid #2d5a2d; border-radius: 20px; padding: 60px 40px; text-align: center; position: relative; overflow: hidden; margin-bottom: 2rem; }
.hero-banner::before { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(ellipse at center, rgba(74,222,74,0.06) 0%, transparent 60%); animation: pulse 4s ease-in-out infinite; }
@keyframes pulse { 0%,100%{transform:scale(1);opacity:.5} 50%{transform:scale(1.1);opacity:1} }
.hero-title { font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #4ade80, #86efac, #bbf7d0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; line-height: 1.1; }
.hero-sub { font-size: 1.15rem; color: #86efac; margin-top: 12px; font-weight: 300; }
.hero-badge { display: inline-block; background: rgba(74,222,74,0.12); border: 1px solid rgba(74,222,74,0.3); border-radius: 50px; padding: 6px 18px; font-size: 0.8rem; color: #4ade80; margin-bottom: 20px; letter-spacing: 2px; text-transform: uppercase; font-weight: 600; }
.stat-row { display: flex; gap: 16px; margin: 1.5rem 0; }
.stat-card { flex: 1; background: linear-gradient(135deg, #0d1f0d, #111f11); border: 1px solid #1e3d1e; border-radius: 14px; padding: 24px 20px; text-align: center; transition: border-color 0.3s; }
.stat-card:hover { border-color: #4ade80; }
.stat-number { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; color: #4ade80; }
.stat-label { font-size: 0.82rem; color: #6b9e6b; margin-top: 4px; }
.result-card { background: linear-gradient(135deg, #0d2b0d, #0f3a0f); border: 1px solid #4ade80; border-radius: 18px; padding: 30px; text-align: center; margin-top: 1.5rem; position: relative; overflow: hidden; }
.result-card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #4ade80, #86efac, #4ade80); background-size: 200% 100%; animation: shimmer 2s linear infinite; }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
.result-label { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #4ade80; }
.confidence-bar-wrap { background: #1a3a1a; border-radius: 50px; height: 10px; margin: 16px 0 6px; overflow: hidden; }
.confidence-bar { height: 100%; border-radius: 50px; background: linear-gradient(90deg, #4ade80, #86efac); }
.disease-card { background: #0d1f0d; border: 1px solid #1e3d1e; border-radius: 14px; padding: 20px; margin: 10px 0; border-left: 4px solid #4ade80; }
.disease-card h4 { font-family: 'Syne', sans-serif; color: #4ade80; margin: 0 0 8px 0; font-size: 1rem; }
.disease-card p { color: #86efac; font-size: 0.9rem; margin: 0; line-height: 1.6; }
.feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 1.5rem 0; }
.feature-card { background: #0d1f0d; border: 1px solid #1e3d1e; border-radius: 14px; padding: 24px; transition: transform 0.2s, border-color 0.2s; }
.feature-card:hover { transform: translateY(-3px); border-color: #4ade80; }
.feature-icon { font-size: 2rem; margin-bottom: 12px; }
.feature-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #bbf7d0; margin-bottom: 8px; }
.feature-desc { color: #6b9e6b; font-size: 0.88rem; line-height: 1.6; }
.stButton > button { background: linear-gradient(135deg, #16a34a, #15803d) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 14px 28px !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; font-size: 1rem !important; width: 100% !important; }
.stButton > button:hover { background: linear-gradient(135deg, #15803d, #166534) !important; box-shadow: 0 8px 25px rgba(74,222,74,0.25) !important; }
.alert-healthy { background: rgba(74,222,74,0.08); border: 1px solid rgba(74,222,74,0.3); border-radius: 12px; padding: 16px 20px; color: #86efac; font-size: 0.95rem; margin-top: 12px; }
.alert-disease { background: rgba(251,146,60,0.08); border: 1px solid rgba(251,146,60,0.3); border-radius: 12px; padding: 16px 20px; color: #fed7aa; font-size: 0.95rem; margin-top: 12px; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #bbf7d0; margin: 2rem 0 1rem 0; padding-bottom: 8px; border-bottom: 1px solid #1e3d1e; }
.step-row { display: flex; gap: 12px; margin: 1rem 0; }
.step { flex: 1; text-align: center; }
.step-num { width: 36px; height: 36px; background: linear-gradient(135deg, #16a34a, #15803d); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px; font-family: 'Syne', sans-serif; font-weight: 700; color: white; font-size: 0.9rem; }
.step-text { font-size: 0.82rem; color: #86efac; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CLASS NAMES — alphabetical order (matches image_dataset_from_directory)
# 38 classes, indices 0–37
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
      "Cherry - Powdery Mildew",                    #  0
    "Peach - Healthy",                             #  1
    "Apple - Cedar Apple Rust",                    #  2
    "Cherry - Healthy",                            #  3
    "Potato - Early Blight",                       #  4
    "Strawberry - Healthy",                        #  5
    "Potato - Late Blight",                        #  6
    "Blueberry - Healthy",                         #  7
    "Tomato - Yellow Leaf Curl Virus",             #  8
    "Tomato - Spider Mites (Two-spotted)",         #  9
    "Orange - Haunglongbing (Citrus Greening)",    # 10
    "Grape - Leaf Blight (Isariopsis)",            # 11
    "Tomato - Bacterial Spot",                     # 12
    "Pepper Bell - Bacterial Spot",                # 13
    "Apple - Healthy",                             # 14
    "Grape - Healthy",                             # 15
    "Tomato - Septoria Leaf Spot",                 # 16
    "Tomato - Late Blight",                        # 17
    "Tomato - Target Spot",                        # 18
    "Pepper Bell - Healthy",                       # 19
    "Apple - Black Rot",                           # 20
    "Tomato - Healthy",                            # 21
    "Corn - Cercospora / Gray Leaf Spot",          # 22
    "Potato - Healthy",                            # 23
    "Corn - Northern Leaf Blight",                 # 24
    "Squash - Powdery Mildew",                     # 25
    "Corn - Common Rust",                          # 26
    "Tomato - Early Blight",                       # 27
    "Grape - Esca (Black Measles)",                # 28
    "Strawberry - Leaf Scorch",                    # 29
    "Corn - Healthy",                              # 30
    "Tomato - Leaf Mold",                          # 31
    "Apple - Apple Scab",                          # 32
    "Peach - Bacterial Spot",                      # 33
    "Raspberry - Healthy",                         # 34
    "Tomato - Mosaic Virus",                       # 35
    "Soybean - Healthy",                           # 36
    "Grape - Black Rot",                           # 37
]

# ─────────────────────────────────────────────────────────────
# MODEL LOADING — tries Keras 3 first, falls back to tf.keras
# model.keras must be in the SAME folder as main.py
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.keras")

    e1 = None
    e2 = None

    # Method 1: keras
    try:
        import keras as _keras
        model = _keras.models.load_model(model_path, compile=False)
        return model
    except Exception as err1:
        e1 = err1

    # Method 2: tf.keras
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as err2:
        e2 = err2

    # If both fail
    st.error(f"❌ Could not load model\n\nMethod 1 error:\n{e1}\n\nMethod 2 error:\n{e2}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# PREDICTION
# resize (256,256) → array → /255.0 → expand_dims → predict
# ─────────────────────────────────────────────────────────────
def predict(image_file):
    mdl = load_model()
    image_file.seek(0) 
    img = Image.open(image_file).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = mdl.predict(arr, verbose=0)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    return idx, confidence, preds[0]
def format_label(raw):
    parts = raw.split("___")
    plant = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return plant, condition

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
        <div style='font-size:2.5rem'>🌿</div>
        <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:#4ade80'>PlantGuard AI</div>
        <div style='font-size:0.78rem;color:#6b9e6b;margin-top:4px'>Disease Detection System</div>
    </div>
    <hr style='border-color:#1e3d1e;margin:16px 0'/>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation",
        ["🏠 Home", "🔍 Diagnose Plant", "📊 Disease Library", "ℹ️ About"],
        label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#1e3d1e;margin:16px 0'/>
    <div style='font-size:0.8rem;color:#4a7a4a;text-align:center;padding-bottom:10px'>
        <div style='color:#4ade80'>✅ 38 Classes Loaded</div>
        <div style='margin-top:6px'>Powered by TensorFlow</div>
        <div style='margin-top:6px'>70,000+ Training Images</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-badge'>🤖 AI-Powered Agriculture</div>
        <div class='hero-title'>PlantGuard AI</div>
        <div class='hero-sub'>Detect plant diseases instantly with deep learning.<br>Protect your crops before it's too late.</div>
    </div>
    <div class='stat-row'>
        <div class='stat-card'><div class='stat-number'>38</div><div class='stat-label'>Disease Classes</div></div>
        <div class='stat-card'><div class='stat-number'>70K+</div><div class='stat-label'>Training Images</div></div>
        <div class='stat-card'><div class='stat-number'>256px</div><div class='stat-label'>Input Resolution</div></div>
        <div class='stat-card'><div class='stat-number'>14</div><div class='stat-label'>Plant Species</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>⚙️ How It Works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='step-row'>
        <div class='step'><div class='step-num'>1</div><div class='step-text'>Upload a leaf image</div></div>
        <div class='step'><div class='step-num'>2</div><div class='step-text'>AI analyses the image</div></div>
        <div class='step'><div class='step-num'>3</div><div class='step-text'>Get instant diagnosis</div></div>
        <div class='step'><div class='step-num'>4</div><div class='step-text'>Follow treatment advice</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🚀 Features</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='feature-grid'>
        <div class='feature-card'><div class='feature-icon'>⚡</div><div class='feature-title'>Instant Detection</div><div class='feature-desc'>Results in under 2 seconds using an optimized CNN.</div></div>
        <div class='feature-card'><div class='feature-icon'>🎯</div><div class='feature-title'>High Accuracy</div><div class='feature-desc'>Trained on 70,000+ real-world plant disease images.</div></div>
        <div class='feature-card'><div class='feature-icon'>💊</div><div class='feature-title'>Treatment Advice</div><div class='feature-desc'>Actionable recommendations for each detected disease.</div></div>
        <div class='feature-card'><div class='feature-icon'>🌱</div><div class='feature-title'>14 Plant Species</div><div class='feature-desc'>Apple, Tomato, Potato, Grape, Corn and many more.</div></div>
        <div class='feature-card'><div class='feature-icon'>📊</div><div class='feature-title'>Confidence Score</div><div class='feature-desc'>See prediction confidence for every result.</div></div>
        <div class='feature-card'><div class='feature-icon'>🔒</div><div class='feature-title'>Private & Secure</div><div class='feature-desc'>Your images are never stored or shared.</div></div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DIAGNOSE
# ============================================================
elif page == "🔍 Diagnose Plant":
    st.markdown("<div class='hero-title' style='font-size:2rem;text-align:left;margin-bottom:0.5rem'>🔍 Plant Disease Diagnosis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b9e6b;margin-bottom:2rem'>Upload a clear image of a plant leaf to get an instant AI diagnosis.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div style='font-family:Syne,sans-serif;font-weight:600;color:#86efac;margin-bottom:8px'>📤 Upload Leaf Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"], label_visibility="collapsed")

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"""
            <div style='margin-top:8px'>
                <span style='background:rgba(74,222,74,0.1);border:1px solid rgba(74,222,74,0.25);border-radius:8px;padding:6px 12px;font-size:0.82rem;color:#86efac;margin-right:6px'>
                📁 {uploaded_file.name}</span>
                <span style='background:rgba(74,222,74,0.1);border:1px solid rgba(74,222,74,0.25);border-radius:8px;padding:6px 12px;font-size:0.82rem;color:#86efac'>
                📦 {round(uploaded_file.size/1024,1)} KB</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Analyse Plant", disabled=(uploaded_file is None))

        st.markdown("""
        <div style='margin-top:20px;padding:16px;background:#0d1f0d;border-radius:12px;border:1px solid #1e3d1e'>
            <div style='font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;color:#4ade80;margin-bottom:10px'>📸 Tips for Best Results</div>
            <div style='font-size:0.85rem;color:#6b9e6b;line-height:1.8'>
            ✓ Use clear, well-lit images<br>
            ✓ Focus on a single leaf<br>
            ✓ Avoid blurry or dark photos<br>
            ✓ Show the full leaf surface
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        if predict_btn and uploaded_file:
            progress = st.progress(0)
            status = st.empty()

            status.markdown("<p style='color:#86efac'>🔬 Preprocessing image...</p>", unsafe_allow_html=True)
            for i in range(40):
                time.sleep(0.008)
                progress.progress(i)

            status.markdown("<p style='color:#86efac'>🧠 Running neural network...</p>", unsafe_allow_html=True)

            try:
                idx, confidence, all_preds = predict(uploaded_file)

                if idx < 0 or idx >= len(CLASS_NAMES):
                    st.error(f"❌ Unexpected prediction index: {idx}. Expected 0–{len(CLASS_NAMES)-1}.")
                    st.stop()

                label     = CLASS_NAMES[idx]
                plant, condition = format_label(label)
                is_healthy = "healthy" in label.lower()

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.stop()

            for i in range(40, 100):
                time.sleep(0.004)
                progress.progress(i)

            status.markdown("<p style='color:#4ade80'>✅ Analysis complete!</p>", unsafe_allow_html=True)
            time.sleep(0.3)
            progress.empty()
            status.empty()

            # Result card
            st.markdown(f"""
            <div class='result-card'>
                <div style='font-size:0.8rem;color:#6b9e6b;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px'>Diagnosis Result</div>
                <div class='result-label'>🌿 {plant}</div>
                <div style='font-size:1.2rem;color:#bbf7d0;margin-top:6px;font-weight:500'>{condition}</div>
                <div style='margin-top:20px'>
                    <div style='display:flex;justify-content:space-between;font-size:0.85rem;color:#6b9e6b;margin-bottom:6px'>
                        <span>Confidence</span>
                        <span style='color:#4ade80;font-weight:600'>{confidence:.1f}%</span>
                    </div>
                    <div class='confidence-bar-wrap'>
                        <div class='confidence-bar' style='width:{confidence}%'></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if is_healthy:
                st.markdown("""
                <div class='alert-healthy'>
                    ✅ <strong>Great news!</strong> Your plant appears healthy. Keep up regular watering and proper sunlight.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='alert-disease'>
                    ⚠️ <strong>Action Required!</strong> <b>{condition}</b> detected on <b>{plant}</b>. Take immediate action to prevent spread.
                </div>""", unsafe_allow_html=True)

            # Top 3
            st.markdown("<div class='section-title' style='font-size:1.1rem;margin-top:1.5rem'>📊 Top 3 Predictions</div>", unsafe_allow_html=True)
            for rank_idx in np.argsort(all_preds)[-3:][::-1]:
                p, c = format_label(CLASS_NAMES[rank_idx])
                pct = float(all_preds[rank_idx]) * 100
                st.markdown(f"""
                <div style='margin-bottom:12px'>
                    <div style='display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px'>
                        <span style='color:#86efac'>{p} — {c}</span>
                        <span style='color:#4ade80;font-weight:600'>{pct:.1f}%</span>
                    </div>
                    <div class='confidence-bar-wrap' style='height:6px'>
                        <div class='confidence-bar' style='width:{pct}%'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        elif not uploaded_file:
            st.markdown("""
            <div style='height:420px;display:flex;flex-direction:column;align-items:center;justify-content:center;
                        background:#0d1f0d;border:2px dashed #1e3d1e;border-radius:18px'>
                <div style='font-size:4rem'>🌿</div>
                <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#4a7a4a;margin-top:16px'>Upload an image to begin</div>
                <div style='font-size:0.85rem;color:#2d5a2d;margin-top:8px'>Supports JPG, JPEG, PNG</div>
            </div>""", unsafe_allow_html=True)

# ============================================================
# DISEASE LIBRARY
# ============================================================
elif page == "📊 Disease Library":
    st.markdown("<div class='hero-title' style='font-size:2rem;text-align:left'>📊 Disease Library</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b9e6b;margin-bottom:2rem'>All 38 plant disease classes the model can detect.</p>", unsafe_allow_html=True)

    plants = {}
    for cls in CLASS_NAMES:
        parts = cls.split("___")
        plant = parts[0].replace("_", " ")
        cond  = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        plants.setdefault(plant, []).append(cond)

    for plant, conditions in plants.items():
        with st.expander(f"🌱 {plant}  ({len(conditions)} classes)"):
            cols = st.columns(2)
            for i, cond in enumerate(conditions):
                with cols[i % 2]:
                    color = "#4ade80" if "healthy" in cond.lower() else "#fb923c"
                    icon  = "✅" if "healthy" in cond.lower() else "⚠️"
                    st.markdown(f"""
                    <div style='background:#0d1f0d;border:1px solid #1e3d1e;border-left:3px solid {color};
                                border-radius:8px;padding:10px 14px;margin:4px 0;font-size:0.88rem;color:#86efac'>
                        {icon} {cond}
                    </div>""", unsafe_allow_html=True)

# ============================================================
# ABOUT
# ============================================================
elif page == "ℹ️ About":
    st.markdown("<div class='hero-title' style='font-size:2rem;text-align:left'>ℹ️ About PlantGuard AI</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='disease-card' style='margin-top:1.5rem'>
        <h4>🎯 Project Goal</h4>
        <p>PlantGuard AI helps farmers and agronomists identify plant diseases early, enabling faster intervention and reducing crop losses.</p>
    </div>
    <div class='disease-card'>
        <h4>🧠 Model Architecture</h4>
        <p>5-block CNN: Conv2D (16→32→64→64→64) + BatchNormalization + MaxPooling, followed by GlobalAveragePooling2D → Dense(128) → Dense(38, softmax).</p>
    </div>
    <div class='disease-card'>
        <h4>⚙️ Training Details</h4>
        <p>Optimizer: Adam (lr=0.0005) | Loss: Categorical Crossentropy | Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau | Input: 256×256 RGB</p>
    </div>
    <div class='disease-card'>
        <h4>📦 Dataset</h4>
        <p>New Plant Diseases Dataset (PlantVillage) — 87,000+ images across 38 classes, 14 plant species. Augmentation: RandomRotation, RandomBrightness, RandomContrast.</p>
    </div>
    <div class='disease-card'>
        <h4>🛠️ Tech Stack</h4>
        <p>Python • TensorFlow • Keras • Streamlit • NumPy • Pillow</p>
    </div>
    <div style='margin-top:2rem;padding:24px;background:#0d1f0d;border-radius:14px;border:1px solid #1e3d1e;text-align:center'>
        <div style='font-family:Syne,sans-serif;font-size:1rem;color:#4ade80;font-weight:700'>🌿 PlantGuard AI</div>
        <div style='font-size:0.85rem;color:#6b9e6b;margin-top:8px'>Built with TensorFlow & Streamlit<br>For educational and agricultural assistance</div>
    </div>
    """, unsafe_allow_html=True)
