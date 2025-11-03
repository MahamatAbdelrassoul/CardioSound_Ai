import streamlit as st
import torch
import numpy as np
from PIL import Image
import librosa
import io
import tempfile
import os
from model import load_model, preprocess_audio, create_spectrogram_plot, predict_heart_condition

# Page configuration
st.set_page_config(
    page_title="CardioSound AI - Heart Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e63946;
        text-align: center;
        margin-bottom: 2rem;
    }
    .clinical-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0077b6;
        margin: 1rem 0;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f8f9fa;
        border-left: 5px solid #e63946;
    }
    .normal-result {
        border-left-color: #2a9d8f !important;
        background-color: #f0f8f4 !important;
    }
    .warning {
        color: #e63946;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .bmi-normal { color: #2a9d8f; font-weight: bold; }
    .bmi-underweight { color: #0077b6; font-weight: bold; }
    .bmi-overweight { color: #e9c46a; font-weight: bold; }
    .bmi-obese { color: #e63946; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_model():
    """Load the trained model with cache"""
    try:
        model = load_model('stgnn_spectrogram_model.pth')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI from height and weight"""
    if height_cm > 0 and weight_kg > 0:
        height_m = height_cm / 100
        return weight_kg / (height_m ** 2)
    return 0

def get_bmi_classification(bmi):
    """Get BMI classification with color coding"""
    if bmi < 18.5:
        return "Underweight", "bmi-underweight", "blue"
    elif bmi < 25:
        return "Normal", "bmi-normal", "green"
    elif bmi < 30:
        return "Overweight", "bmi-overweight", "orange"
    else:
        return "Obese", "bmi-obese", "red"

def main():
    # Header
    st.markdown('<h1 class="main-header"> CardioSound AI</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Heart Sound Analysis")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Configuration")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This AI-powered application analyzes heart sounds 
    and detects potential cardiac abnormalities.
    
    **Features:**
    - Real-time audio analysis
    - Spectrogram generation
    - Clinical data integration
    - AI-powered detection
    """)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_cached_model()
    
    if model is None:
        st.error("Cannot load model. Please check if the .pth file is present.")
        return
    
    # Main section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Patient Clinical Information")
        
        # Clinical data form
        with st.form("clinical_data"):
            st.markdown('<div class="clinical-section">', unsafe_allow_html=True)
            
            # Age category
            age_options = {
                0: "👶 Neonate (0-1 month)",
                1: "🧒 Infant (1-12 months)", 
                2: "👦 Child (1-12 years)",
                3: "👨 Adolescent (13-18 years)"
            }
            age_selected = st.selectbox(
                "Age Category",
                options=list(age_options.keys()),
                format_func=lambda x: age_options[x],
                help="Select the patient's age category"
            )
            
            # Sex
            sex = st.radio(
                "Sex",
                options=["Female", "Male"],
                horizontal=True,
                help="Patient's biological sex"
            )
            
            # Height and Weight
            col_hw1, col_hw2 = st.columns(2)
            with col_hw1:
                height = st.number_input(
                    "Height (cm)",
                    min_value=0,
                    max_value=200,
                    value=170,
                    help="Patient height in centimeters"
                )
            with col_hw2:
                weight = st.number_input(
                    "Weight (kg)", 
                    min_value=0,
                    max_value=150,
                    value=70,
                    help="Patient weight in kilograms"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Audio upload section
            st.subheader("🎵 Heart Sound Recording")
            
            uploaded_file = st.file_uploader(
                "Upload heart sound recording",
                type=['wav', 'mp3', 'm4a', 'ogg'],
                help="Recommended: WAV format, 22050Hz, 3-5 seconds duration"
            )
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Analyze Heart Sound", 
                type="primary",
                use_container_width=True
            )
    
    with col2:
        st.subheader("📋 Instructions")
        st.info("""
        **For best results:**
        
        1. **Clinical Data:** Provide accurate patient information
        2. **Recording Quality:** Use digital stethoscope if possible
        3. **Environment:** Record in quiet setting
        4. **Duration:** 3-5 seconds ideal
        5. **Format:** WAV recommended, 22050Hz sampling rate
        
        **Auscultation Locations:**
        - Aortic Valve (AV)
        - Pulmonary Valve (PV) 
        - Tricuspid Valve (TV)
        - Mitral Valve (MV)
        """)
        
        # BMI Calculation and Display (OUTSIDE the form for real-time updates)
        st.markdown("### 📊 Clinical Insights")
        
        # Calculate BMI
        bmi = calculate_bmi(height, weight)
        bmi_status, bmi_class, bmi_color = get_bmi_classification(bmi)
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Age Category", age_options[age_selected].split(" ")[1])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_stat2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Sex", sex)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_stat3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("BMI", f"{bmi:.1f}")
            st.markdown(f'<p class="{bmi_class}">{bmi_status}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # BMI Explanation
        if bmi > 0:
            st.markdown(f"""
            **BMI Analysis:**
            - **Value:** {bmi:.1f}
            - **Classification:** <span class="{bmi_class}">{bmi_status}</span>
            - **Health Risk:** {"Low" if bmi_status == "Normal" else "Increased"}
            """, unsafe_allow_html=True)
    
    # Process analysis when form is submitted
    if submitted and uploaded_file is not None:
        with st.spinner("Analyzing heart sound... Generating spectrograms..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                # Load and verify audio
                audio_data, sr = librosa.load(audio_path, sr=22050)
                st.success(f"✅ Audio loaded: {len(audio_data)} samples at {sr}Hz")
                
                # Display audio player
                st.audio(uploaded_file)
                
                # Prepare clinical data for model
                clinical_data = {
                    'age_encoded': age_selected,
                    'sex_encoded': 1 if sex == "Male" else 0,
                    'height': height,
                    'weight': weight, 
                    'bmi': bmi
                }
                
                # Preprocess audio with clinical data
                audio_tensor, clinical_tensor, processed_audio, processed_sr = preprocess_audio(
                    audio_path, clinical_data
                )
                
                if audio_tensor is not None:
                    # Make prediction
                    result = predict_heart_condition(model, audio_tensor, clinical_tensor)
                    
                    # Generate spectrogram visualization
                    spectrogram_image = create_spectrogram_plot(
                        processed_audio, processed_sr, result['spectrograms']
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Results layout
                    res_col1, res_col2 = st.columns([1, 1])
                    
                    with res_col1:
                        # Spectrogram
                        st.image(spectrogram_image, caption="AI-Generated Spectrogram", use_column_width=True)
                    
                    with res_col2:
                        # Classification results
                        prediction = result['prediction']
                        probs = result['probabilities']
                        
                        # Result box with conditional styling
                        result_class = "" if prediction == 1 else "normal-result"
                        st.markdown(f'<div class="result-box {result_class}">', unsafe_allow_html=True)
                        
                        if prediction == 1:
                            st.error("## ⚠️ Result: ABNORMAL")
                            st.warning("**Recommendation:** Medical consultation recommended")
                        else:
                            st.success("## ✅ Result: NORMAL") 
                            st.info("**Recommendation:** Continue regular monitoring")
                        
                        # Detailed probabilities
                        st.markdown("### Probability Breakdown")
                        col_prob1, col_prob2 = st.columns(2)
                        
                        with col_prob1:
                            st.metric(
                                label="Normal Probability",
                                value=f"{probs[0]*100:.1f}%",
                                delta="Low risk" if probs[0] > 0.7 else ""
                            )
                        
                        with col_prob2:
                            st.metric(
                                label="Abnormal Probability",
                                value=f"{probs[1]*100:.1f}%", 
                                delta="High risk" if probs[1] > 0.7 else "",
                                delta_color="inverse"
                            )
                        
                        # Risk indicator
                        st.progress(float(probs[1]), text="Abnormality Detection Confidence")
                        
                        # Clinical context
                        st.markdown("### 🩺 Clinical Context")
                        st.write(f"**Age:** {age_options[age_selected]}")
                        st.write(f"**Sex:** {sex}")
                        st.write(f"**BMI:** {bmi:.1f} ({bmi_status})")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Cleanup
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
                        
                else:
                    st.error("Error during audio preprocessing")
                    
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                # Cleanup on error
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.unlink(audio_path)
    
    elif submitted and uploaded_file is None:
        st.error("❌ Please upload a heart sound recording for analysis")
    
    else:
        # Welcome state
        st.markdown("---")
        st.info("👆 Please fill in patient clinical information and upload a heart sound recording to begin analysis")
        
        # Example format
        with st.expander("🎧 Recommended Audio Format"):
            st.write("""
            **Ideal Specifications:**
            - Format: WAV (uncompressed)
            - Sample Rate: 22050 Hz
            - Channels: Mono (1 channel) 
            - Duration: 3-5 seconds
            - Audio Level: -3dB to -6dB (avoid clipping)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "CardioSound AI - Intelligent Cardiac Analysis System<br>"
        "⚠️ This tool is intended for healthcare professionals and does not replace comprehensive medical diagnosis"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()