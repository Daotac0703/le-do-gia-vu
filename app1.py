"""
Ứng dụng nhận dạng người vs không phải người
Sinh viên: Lê Quang Đạo - MSSV: 223332821
"""

import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import requests
from io import BytesIO

st.set_page_config(
    page_title="Human Detection AI",
    page_icon="🧠",
    layout="wide"
)

# ==================== CUSTOM DESIGN ====================
st.markdown("""
<style>
    :root {
        --primary: #FF6B6B;
        --secondary: #4ECDC4;
        --accent: #FFE66D;
        --dark: #2D3436;
        --light: #F7F9FC;
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #F7F9FC 0%, #E8F0F5 100%);
    }
    
    .main {
        background: transparent;
    }
    
    .top-section {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E72 50%, #FFB84D 100%);
        padding: 50px 30px;
        border-radius: 0px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.2);
        margin-bottom: 30px;
    }
    
    .top-section h1 {
        font-size: 2.8em;
        margin: 0 0 10px 0;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    .top-section p {
        font-size: 1.1em;
        opacity: 0.95;
        margin: 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 3px solid #FFE66D;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 14px 24px;
        background: white;
        color: #2D3436;
        border-radius: 12px 12px 0 0;
        margin-right: 8px;
        font-weight: 600;
        border-bottom: 3px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4ECDC4 !important;
        color: white !important;
    }
    
    .card-container {
        background: white;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        margin: 20px 0;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .card-container:hover {
        border-color: #4ECDC4;
        box-shadow: 0 12px 32px rgba(78, 205, 196, 0.15);
    }
    
    .result-success {
        background: linear-gradient(135deg, #11D3B3 0%, #4ECDC4 100%);
        color: white;
        padding: 40px;
        border-radius: 16px;
        text-align: center;
        margin: 30px 0;
        font-weight: bold;
        font-size: 1.3em;
        box-shadow: 0 12px 32px rgba(78, 205, 196, 0.3);
    }
    
    .result-danger {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E72 100%);
        color: white;
        padding: 40px;
        border-radius: 16px;
        text-align: center;
        margin: 30px 0;
        font-weight: bold;
        font-size: 1.3em;
        box-shadow: 0 12px 32px rgba(255, 107, 107, 0.3);
    }
    
    .confidence-container {
        margin-top: 20px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 8px;
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
        overflow: hidden;
        margin: 15px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: white;
        border-radius: 4px;
        transition: width 0.6s ease;
    }
    
    .section-title {
        font-size: 1.5em;
        font-weight: 700;
        color: #2D3436;
        margin: 25px 0 15px 0;
        padding-left: 0;
    }
    
    .info-badge {
        display: inline-block;
        background: linear-gradient(135deg, #FFE66D 0%, #FFB84D 100%);
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        color: #2D3436;
        margin: 5px;
    }
    
    .footer-section {
        background: #2D3436;
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-top: 40px;
    }
    
    .guide-item {
        background: white;
        padding: 20px;
        border-radius: 12px;
        margin: 12px 0;
        border-left: 4px solid #4ECDC4;
        line-height: 1.8;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E72 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(255, 107, 107, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="top-section">
    <h1>🧠 HUMAN DETECTION</h1>
    <p>Advanced Deep Learning Recognition</p>
</div>
""", unsafe_allow_html=True)

# ==================== INFO ====================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<span class="info-badge">👨‍💼 Lê Quang Đạo</span>', unsafe_allow_html=True)
with col2:
    st.markdown('<span class="info-badge">🎓 223332821</span>', unsafe_allow_html=True)
with col3:
    st.markdown('<span class="info-badge">📚 CNN Model</span>', unsafe_allow_html=True)

st.divider()

# ==================== CONFIG ====================
IMG_SIZE = 64

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('humantachi.h5')
        return model
    except:
        return None

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image):
    img_array = preprocess_image(image)
    return model.predict(img_array, verbose=0)[0][0]

def show_result(confidence, is_human):
    if is_human:
        st.markdown(f"""
        <div class="result-success">
            ✅ PERSON DETECTED<br>
            <div class="confidence-container">
                <small>Confidence: {confidence:.1f}%</small>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {100-confidence}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-danger">
            ❌ NOT A PERSON<br>
            <div class="confidence-container">
                <small>Confidence: {confidence:.1f}%</small>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {confidence}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN ====================
model = load_model()

if model is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "📸 Webcam", "🌐 URL", "📘 Guide"])
    
    # TAB 1: Upload
    with tab1:
        st.markdown('<div class="section-title">📁 Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose image", type=['jpg','jpeg','png','bmp','webp'], key='upload1')
        if uploaded:
            col1, col2 = st.columns(2)
            with col1:
                img = Image.open(uploaded)
                st.image(img, use_container_width=True)
            with col2:
                st.write("")
                if st.button("🔍 Analyze", use_container_width=True, key='btn1'):
                    with st.spinner("Processing..."):
                        pred = predict(model, img)
                        conf = pred * 100 if pred > 0.5 else (1-pred) * 100
                        show_result(conf, pred <= 0.5)
    
    # TAB 2: Webcam
    with tab2:
        st.markdown('<div class="section-title">📸 Webcam</div>', unsafe_allow_html=True)
        pic = st.camera_input("Take a photo")
        if pic:
            col1, col2 = st.columns(2)
            with col1:
                img = Image.open(pic)
                st.image(img, use_container_width=True)
            with col2:
                st.write("")
                if st.button("🔍 Analyze", use_container_width=True, key='btn2'):
                    with st.spinner("Processing..."):
                        pred = predict(model, img)
                        conf = pred * 100 if pred > 0.5 else (1-pred) * 100
                        show_result(conf, pred <= 0.5)
    
    # TAB 3: URL
    with tab3:
        st.markdown('<div class="section-title">🌐 Image URL</div>', unsafe_allow_html=True)
        url = st.text_input("Paste image URL", placeholder="https://...")
        if url and st.button("Load & Analyze", use_container_width=True):
            try:
                with st.spinner("Loading..."):
                    resp = requests.get(url, timeout=10)
                    img = Image.open(BytesIO(resp.content))
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, use_container_width=True)
                with col2:
                    with st.spinner("Processing..."):
                        pred = predict(model, img)
                        conf = pred * 100 if pred > 0.5 else (1-pred) * 100
                        show_result(conf, pred <= 0.5)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # TAB 4: Guide
    with tab4:
        st.markdown('<div class="section-title">📘 User Guide</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="guide-item">
        <b>✨ Features</b><br>
        • Upload images from your device<br>
        • Capture photos with webcam<br>
        • Analyze images from URLs
        </div>
        
        <div class="guide-item">
        <b>📋 Formats Supported</b><br>
        JPG • JPEG • PNG • BMP • WEBP
        </div>
        
        <div class="guide-item">
        <b>🎯 How to Use</b><br>
        1. Select a tab (Upload, Webcam, or URL)<br>
        2. Provide image input<br>
        3. Click Analyze<br>
        4. View results
        </div>
        
        <div class="guide-item">
        <b>⚙️ Model Info</b><br>
        • Type: CNN (Convolutional Neural Network)<br>
        • Input: 64x64 pixels<br>
        • Classes: Person / Non-Person<br>
        • File: humantachi.h5
        </div>
        
        <div class="guide-item">
        <b>💡 Tips</b><br>
        • Clear images = Better results<br>
        • Avoid blurry photos<br>
        • Confidence > 50% = Non-Person<br>
        • Confidence < 50% = Person
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card-container" style="border-left: 4px solid #FF6B6B;">
    <h3>❌ Model Not Found</h3>
    <p>Place <b>humantachi.h5</b> in the same directory as this app</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer-section">
    <p style="margin: 0; font-size: 1.1em; font-weight: 600;">🎓 Human Detection System</p>
    <p style="margin: 5px 0; opacity: 0.8;">© 2026 Lê Quang Đạo | TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)



