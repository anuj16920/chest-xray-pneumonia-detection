"""
Chest X-Ray Pneumonia Detection - Web Interface
Streamlit web app for easy prediction
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

# Page config
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .pneumonia-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pneumonia_model():
    """Load the trained model"""
    model_path = 'models/best_model.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        return None

def preprocess_image(uploaded_image):
    """Preprocess uploaded image for prediction"""
    # Convert to RGB
    img = Image.open(uploaded_image).convert('RGB')
    
    # Resize to 224x224
    img = img.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img, img_array

def predict(model, img_array):
    """Make prediction"""
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    if prediction > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = prediction
    else:
        diagnosis = "NORMAL"
        confidence = 1 - prediction
    
    return diagnosis, confidence, prediction

def main():
    # Header
    st.markdown('<p class="main-header">🩺 Chest X-Ray Pneumonia Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Medical Image Analysis</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_pneumonia_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first.")
        st.info("Run `python src/04_train_model.py` to train the model.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.header("📋 About")
    st.sidebar.info("""
    This AI model detects pneumonia from chest X-ray images using 
    deep learning (Transfer Learning with ResNet50).
    
    **How to use:**
    1. Upload a chest X-ray image
    2. Click 'Analyze X-Ray'
    3. Get instant diagnosis
    
    **Disclaimer:** This is for educational purposes only. 
    Always consult a healthcare professional for medical diagnosis.
    """)
    
    st.sidebar.header("📊 Model Info")
    st.sidebar.write("**Architecture:** ResNet50 (Transfer Learning)")
    st.sidebar.write("**Training Data:** Chest X-Ray Images (Pneumonia)")
    st.sidebar.write("**Classes:** Normal, Pneumonia")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                with st.spinner("Analyzing X-ray image... Please wait."):
                    # Preprocess
                    img_display, img_array = preprocess_image(uploaded_file)
                    
                    # Predict
                    diagnosis, confidence, raw_pred = predict(model, img_array)
                    
                    # Store in session state
                    st.session_state.diagnosis = diagnosis
                    st.session_state.confidence = confidence
                    st.session_state.raw_pred = raw_pred
    
    with col2:
        st.header("📊 Diagnosis Result")
        
        if 'diagnosis' in st.session_state:
            diagnosis = st.session_state.diagnosis
            confidence = st.session_state.confidence
            raw_pred = st.session_state.raw_pred
            
            # Prediction box
            if diagnosis == "PNEUMONIA":
                st.markdown(f"""
                <div class="prediction-box pneumonia-box">
                    <h2 style="color: #dc3545; margin: 0;">⚠️ PNEUMONIA DETECTED</h2>
                    <h3 style="color: #721c24; margin-top: 1rem;">
                        Confidence: {confidence*100:.2f}%
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.warning("**Recommendation:** Please consult a healthcare professional immediately for proper diagnosis and treatment.")
                
            else:
                st.markdown(f"""
                <div class="prediction-box normal-box">
                    <h2 style="color: #28a745; margin: 0;">✅ NORMAL</h2>
                    <h3 style="color: #155724; margin-top: 1rem;">
                        Confidence: {confidence*100:.2f}%
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("**Result:** No signs of pneumonia detected in this X-ray.")
            
            # Confidence meter
            st.subheader("Confidence Score")
            st.progress(float(confidence))
            st.write(f"Model confidence: **{confidence*100:.2f}%**")
            
            # Technical details (expandable)
            with st.expander("🔧 Technical Details"):
                st.write(f"**Raw Prediction Score:** {raw_pred:.4f}")
                st.write(f"**Classification Threshold:** 0.5")
                st.write(f"**Predicted Class:** {diagnosis}")
                
                if diagnosis == "PNEUMONIA":
                    st.write(f"**Interpretation:** Score > 0.5 indicates PNEUMONIA")
                else:
                    st.write(f"**Interpretation:** Score ≤ 0.5 indicates NORMAL")
            
            # Download report
            if st.button("📥 Download Report", use_container_width=True):
                report = f"""
                CHEST X-RAY PNEUMONIA DETECTION REPORT
                =====================================
                
                Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                DIAGNOSIS: {diagnosis}
                Confidence: {confidence*100:.2f}%
                Raw Score: {raw_pred:.4f}
                
                {("⚠️ PNEUMONIA DETECTED - Please consult a healthcare professional" 
                  if diagnosis == "PNEUMONIA" 
                  else "✅ NORMAL - No pneumonia detected")}
                
                =====================================
                Disclaimer: This is an AI-based analysis for educational 
                purposes only. Always consult a healthcare professional 
                for medical diagnosis and treatment.
                """
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"xray_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("👆 Upload an X-ray image and click 'Analyze X-Ray' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d;">
        <p><strong>⚠️ Medical Disclaimer</strong></p>
        <p>This tool is for educational and research purposes only. 
        It should not be used as a substitute for professional medical advice, 
        diagnosis, or treatment. Always seek the advice of a qualified healthcare 
        provider with any questions regarding a medical condition.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    import pandas as pd
    main()
