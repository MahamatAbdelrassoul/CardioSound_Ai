import streamlit as st
import torch
import numpy as np
from PIL import Image
import librosa
import io
import tempfile
import os
from model import load_model, preprocess_audio, create_spectrogram_plot, predict_heart_condition

# Futuristic page configuration
st.set_page_config(
    page_title="CardioSound AI - Neural Heart Analysis",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Futuristic CSS - SCI-FI DESIGN
st.markdown("""
<style>
    /* FUTURISTIC NEON COLORS */
    :root {
        --neon-blue: #00f5ff;
        --neon-purple: #bc13fe;
        --neon-green: #39ff14;
        --neon-red: #ff073a;
        --matrix-green: #00ff41;
        --cyber-dark: #0a0a0a;
        --cyber-surface: #1a1a1a;
        --cyber-border: #333333;
    }
    
    .main {
        background-color: var(--cyber-dark);
        color: var(--neon-green);
    }
    
    .cyber-header {
        background: linear-gradient(135deg, var(--cyber-dark) 0%, #1a1a2e 100%);
        color: var(--neon-blue);
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 3px solid var(--neon-purple);
        position: relative;
        overflow: hidden;
    }
    
    .cyber-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--neon-blue), transparent);
        animation: scan 3s linear infinite;
    }
    
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .neon-card {
        background: var(--cyber-surface);
        padding: 2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid var(--cyber-border);
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .neon-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--neon-blue), var(--neon-purple));
    }
    
    .neon-card:hover {
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.3);
        transform: translateY(-2px);
    }
    
    .cyber-metric {
        background: var(--cyber-surface);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--cyber-border);
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .cyber-metric:hover {
        box-shadow: 0 0 25px rgba(0, 245, 255, 0.2);
        border-color: var(--neon-blue);
    }
    
    .analysis-terminal {
        background: var(--cyber-surface);
        padding: 2.5rem;
        border-radius: 8px;
        margin: 2rem 0;
        border: 1px solid var(--neon-green);
        box-shadow: 0 0 30px rgba(57, 255, 20, 0.2);
        font-family: 'Courier New', monospace;
    }
    
    .result-neon {
        background: var(--cyber-surface);
        padding: 2.5rem;
        border-radius: 8px;
        margin: 2rem 0;
        border: 1px solid var(--neon-green);
        box-shadow: 0 0 30px rgba(57, 255, 20, 0.3);
    }
    
    .result-alert {
        border: 1px solid var(--neon-red);
        box-shadow: 0 0 30px rgba(255, 7, 58, 0.3);
    }
    
    /* Cyber buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--neon-blue) 0%, var(--neon-purple) 100%);
        color: var(--cyber-dark) !important;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 4px;
        font-weight: 700;
        font-size: 1.1rem;
        font-family: 'Courier New', monospace;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.4);
    }
    
    .stButton button:hover {
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.6);
        transform: translateY(-2px);
    }
    
    /* Cyber inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: var(--cyber-surface) !important;
        color: var(--neon-green) !important;
        border: 1px solid var(--cyber-border) !important;
        border-radius: 4px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: var(--neon-blue) !important;
        box-shadow: 0 0 10px rgba(0, 245, 255, 0.3) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--cyber-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--neon-blue);
        border-radius: 4px;
    }
    
    /* Streamlit specific overrides */
    .stAlert {
        background-color: var(--cyber-surface) !important;
        border: 1px solid var(--cyber-border) !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--neon-blue), var(--neon-purple)) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_model():
    """Load the trained model with cache"""
    try:
        model = load_model('stgnn_spectrogram_model.pth')
        return model
    except Exception as e:
        st.error(f"🔄 SYSTEM ERROR: Neural network failed to initialize")
        return None

def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI from height and weight"""
    if height_cm > 0 and weight_kg > 0:
        height_m = height_cm / 100
        return weight_kg / (height_m ** 2)
    return 0

def get_bmi_classification(bmi):
    """Get BMI classification with cyber styling"""
    if bmi < 18.5:
        return "UNDERWEIGHT", "#00f5ff", "⚡"
    elif bmi < 25:
        return "OPTIMAL", "#39ff14", "✅"
    elif bmi < 30:
        return "ELEVATED", "#ffaa00", "⚠️"
    else:
        return "CRITICAL", "#ff073a", "🚨"

def main():
    # CYBER HEADER
    st.markdown("""
    <div class="cyber-header">
        <h1 style="margin:0; font-size: 3rem; font-weight: 700; text-shadow: 0 0 10px var(--neon-blue);">🌟 CARDIOSOUND AI</h1>
        <p style="margin:0; font-size: 1.2rem; color: var(--neon-green); margin-top: 0.5rem; font-family: 'Courier New', monospace;">
            NEURAL HEART ANALYSIS SYSTEM // ONLINE
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cyber sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem; border: 1px solid var(--cyber-border); border-radius: 8px; margin-bottom: 1rem;">
            <h2 style="color: var(--neon-blue); margin:0; text-shadow: 0 0 10px var(--neon-blue);">🔮</h2>
            <h3 style="margin:0; color: var(--neon-green); font-family: 'Courier New', monospace;">CARDIO AI</h3>
            <p style="font-size: 0.8rem; color: var(--neon-blue); margin:0; font-family: 'Courier New', monospace;"></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Welcome to Our Neural Heart Analysis System")
        st.success("""
        A smarter way to detect and predict heart failures using advanced AI technology.
        """)
        
        st.markdown("---")
        st.markdown("### What It Does")
        st.write("**Detect early signs of heart failure from patient data**")
        st.write("**Provide the Spectrogram of heart sounds for analysis**")
        st.write("**Give the result**")

    # Load model
    with st.spinner("🔄 INITIALIZING NEURAL NETWORK..."):
        model = load_cached_model()
    
    if model is None:
        st.error("🚫 CRITICAL FAILURE: Neural network offline")
        return

    # Cyber main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 👤 BIO-METRIC PROFILE")
        
        with st.form("bio_profile"):
            st.markdown('<div class="neon-card">', unsafe_allow_html=True)
            
            # Cyber age selection
            age_options = {
                0: "👶 NEONATE [0-1M]",
                1: "🧒 INFANT [1-12M]", 
                2: "👦 CHILD [1-12Y]",
                3: "👨 ADOLESCENT [13-18Y]"
            }
            age_selected = st.selectbox(
                "**SELECT AGE PROTOCOL**",
                options=list(age_options.keys()),
                format_func=lambda x: age_options[x]
            )
            
            # Cyber gender selection
            sex = st.radio(
                "**BIOLOGICAL SEX**",
                options=["FEMALE", "MALE"],
                horizontal=True
            )
            
            # Cyber measurements
            st.markdown("#### BODY METRICS")
            col_hw1, col_hw2 = st.columns(2)
            with col_hw1:
                height = st.number_input(
                    "**HEIGHT (CM)**",
                    min_value=0,
                    max_value=200,
                    value=120
                )
            with col_hw2:
                weight = st.number_input(
                    "**WEIGHT (KG)**", 
                    min_value=0,
                    max_value=150,
                    value=25
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Cyber file upload
            st.markdown("### 🎵 HEART SIGNAL INPUT")
            uploaded_file = st.file_uploader(
                "**UPLOAD CARDIAC FREQUENCY DATA**",
                type=['wav', 'mp3', 'm4a', 'ogg'],
                help="NEURAL NET PREFERS: WAV FORMAT, 22050Hz, 3-5s DURATION"
            )
            
            # Cyber submit button
            submitted = st.form_submit_button(
                "🚀 INITIATE NEURAL ANALYSIS", 
                type="primary",
                use_container_width=True
            )
    
    with col2:
        st.markdown("### 📊 SYSTEM METRICS")
        
        # Real-time cyber metrics
        bmi = calculate_bmi(height, weight)
        bmi_status, bmi_color, bmi_icon = get_bmi_classification(bmi)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="cyber-metric">', unsafe_allow_html=True)
            st.metric("AGE PROTOCOL", age_options[age_selected].split(" ")[1])
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="cyber-metric">', unsafe_allow_html=True)
            st.metric("BIOLOGICAL", sex)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="cyber-metric">', unsafe_allow_html=True)
            st.metric("BMI INDEX", f"{bmi:.1f}")
            st.markdown(f'<p style="color: {bmi_color}; font-weight: 700; margin:0; font-family: Courier New;">{bmi_icon} {bmi_status}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🔮 ANALYSIS PROTOCOL")
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        st.info("""
        **NEURAL PROCESSING PIPELINE:**
        1. **INPUT:** Heart frequency data
        2. **PROCESS:** Neural network analysis  
        3. **ANALYZE:** Pattern recognition
        4. **OUTPUT:** Diagnostic probability
        
        **SYSTEM CAPABILITIES:**
        • Real-time signal processing
        • Deep learning algorithms
        • Anomaly detection
        • Predictive analytics
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Cyber analysis process
    if submitted and uploaded_file is not None:
        with st.spinner("🔮 NEURAL NETWORK ANALYZING FREQUENCY PATTERNS..."):
            try:
                # Save and process audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                audio_data, sr = librosa.load(audio_path, sr=22050)
                st.success(f"✅ SIGNAL ACQUIRED: {len(audio_data)} data points")
                
                # Play audio
                st.audio(uploaded_file)
                
                # Prepare data
                clinical_data = {
                    'age_encoded': age_selected,
                    'sex_encoded': 1 if sex == "MALE" else 0,
                    'height': height,
                    'weight': weight, 
                    'bmi': bmi
                }
                
                # Neural analysis
                audio_tensor, clinical_tensor, processed_audio, processed_sr = preprocess_audio(
                    audio_path, clinical_data
                )
                
                if audio_tensor is not None:
                    result = predict_heart_condition(model, audio_tensor, clinical_tensor)
                    spectrogram_image = create_spectrogram_plot(
                        processed_audio, processed_sr, result['spectrograms']
                    )
                    
                    # Cyber results display
                    st.markdown("---")
                    st.markdown("## 📈 NEURAL ANALYSIS COMPLETE")
                    
                    res_col1, res_col2 = st.columns([1, 1])
                    
                    with res_col1:
                        st.image(spectrogram_image, caption="NEURAL FREQUENCY MAP", use_column_width=True)
                    
                    with res_col2:
                        prediction = result['prediction']
                        probs = result['probabilities']
                        
                        result_class = "result-neon" if prediction == 0 else "result-neon result-alert"
                        st.markdown(f'<div class="{result_class}">', unsafe_allow_html=True)
                        
                        if prediction == 1:
                            st.error("## ⚠️ ANOMALY DETECTED")
                            st.warning("""
                            **NEURAL NET ALERT:**
                            • ABNORMAL PATTERN RECOGNIZED
                            • MEDICAL CONSULTATION ADVISED
                            • FURTHER DIAGNOSTICS RECOMMENDED
                            """)
                        else:
                            st.success("## ✅ PATTERN NOMINAL")
                            st.info("""
                            **SYSTEM STATUS:**
                            • HEART PATTERNS WITHIN PARAMETERS
                            • CONTINUE STANDARD MONITORING
                            • SCHEDULE ROUTINE ANALYSIS
                            """)
                        
                        # Cyber probability display
                        st.markdown("### NEURAL CONFIDENCE")
                        col_p1, col_p2 = st.columns(2)
                        
                        with col_p1:
                            st.metric(
                                "NOMINAL",
                                f"{probs[0]*100:.1f}%",
                                "HIGH CONFIDENCE" if probs[0] > 0.7 else ""
                            )
                        
                        with col_p2:
                            st.metric(
                                "ANOMALY",
                                f"{probs[1]*100:.1f}%",
                                "REVIEW REQUIRED" if probs[1] > 0.3 else "",
                                delta_color="inverse"
                            )
                        
                        st.progress(float(probs[1]), text="NEURAL DETECTION CERTAINTY")
                        
                        st.markdown("### BIO-PROFILE CONTEXT")
                        st.write(f"**SUBJECT:** {age_options[age_selected]}, {sex}")
                        st.write(f"**BMI STATUS:** {bmi:.1f} [{bmi_status}]")
                        st.write(f"**NEURAL CERTAINTY:** {max(probs)*100:.1f}%")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Cleanup
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
                        
                else:
                    st.error("🔄 PROCESSING FAILURE: Signal quality insufficient")
                    
            except Exception as e:
                st.error(f"🚫 SYSTEM ERROR: Neural analysis interrupted")
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.unlink(audio_path)
    
    elif submitted and uploaded_file is None:
        st.error("📡 NO SIGNAL: Upload heart frequency data")
    
    else:
        # Clinical ready state
        st.markdown("---")
        st.markdown("""
        <div class="analysis-terminal">
            <p style="color: var(--neon-green); font-family: 'Courier New', monospace; margin:0;">
                > SYSTEM READY FOR ANALYSIS<br>
                > AWAITING BIO-METRIC INPUT<br>
                > NEURAL NETWORK: STANDBY<br>
                > SECURITY: ACTIVE<br>
                > READY FOR HEART SIGNAL PROCESSING...
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Cyber footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--neon-blue); font-size: 0.8rem; padding: 2rem; font-family: 'Courier New', monospace;">
        <strong>CARDIOSOUND AI</strong> // NEURAL HEART ANALYSIS SYSTEM <br>
        FOR AUTHORIZED MEDICAL PERSONNEL ONLY // SECURITY CLEARANCE REQUIRED
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()