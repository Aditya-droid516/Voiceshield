import streamlit as st
import requests
import json

st.set_page_config(
    page_title="VoiceShield | Enterprise Deepfake & Vishing Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        width: 100%; border-radius: 8px; font-weight: 600; height: 3em;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white; border: none; transition: 0.3s;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
    .real-badge {
        background-color: rgba(46, 204, 113, 0.15); color: #2ecc71;
        padding: 12px; border-radius: 8px; border: 1px solid #2ecc71;
        font-weight: bold; text-align: center; font-size: 22px;
    }
    .fake-badge {
        background-color: rgba(231, 76, 60, 0.15); color: #e74c3c;
        padding: 12px; border-radius: 8px; border: 1px solid #e74c3c;
        font-weight: bold; text-align: center; font-size: 22px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State for Clearing
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = 0

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-checked.png", width=70)
    st.title("VoiceShield Security")
    st.markdown("---")
    st.info(
        "**Innovative Modules Active:**\n"
        "1. **File Audit & Cryptographic Hashing**\n"
        "2. **Live Vishing (Mic) Guard**\n"
        "3. **Forensic JSON Certificate Generator**"
    )
    st.markdown("### System Status")
    st.success("🟢 Backend API: Online")
    st.markdown("---")
    st.markdown("### Protection Guidelines")
    st.markdown("- Ideal for screening phone call snippets, voice notes, and audio evidence.")

# Main Header
st.title("🛡️ VoiceShield Intelligence Console")
st.subheader("Real-Time AI Voice-Clone Detection & Vishing Defense System")
st.markdown("---")

# Navigation Tabs for Modes
tab_file, tab_mic = st.tabs(["📁 Audio File Screening", "🎙️ Live Vishing Guard (Microphone)"])

audio_file_to_send = None

with tab_file:
    st.markdown("### Upload Audio Evidence")
    uploaded_file = st.file_uploader(
        "Select recording (.wav, .mp3, .flac, .m4a)", 
        type=["wav", "mp3", "flac", "m4a", "webm"],
        key=f"file_uploader_{st.session_state.reset_trigger}"
    )
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        audio_file_to_send = uploaded_file

with tab_mic:
    st.markdown("### Live Call / Vishing Interception")
    st.write("Record a live voice snippet directly from your microphone to test for real-time AI synthesis.")
    
    mic_audio = st.audio_input(
        "Record speech snippet", 
        key=f"mic_input_{st.session_state.reset_trigger}"
    )
    
    if mic_audio is not None:
        st.audio(mic_audio)
        mic_audio.name = "live_vishing_intercept.wav"
        audio_file_to_send = mic_audio

# Delete / Clear Button to remove stored/recorded audio
if audio_file_to_send is not None:
    st.markdown("")
    if st.button("🗑️ Clear / Delete Stored Audio & Reset", type="secondary"):
        st.session_state.reset_trigger += 1
        st.rerun()

st.markdown("---")

# Audit Execution Section
if audio_file_to_send is not None:
    st.markdown("### 🔍 Forensic Security Scan")
    if st.button("Execute Deepfake & Vishing Audit", type="primary"):
        with st.spinner("Analyzing spectral vectors and generating cryptographic audit trail..."):
            try:
                files = {"file": (audio_file_to_send.name, audio_file_to_send.getvalue(), "audio/wav")}
                response = requests.post("http://localhost:8000/analyze", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction", "Unknown")
                    confidence = float(result.get("confidence", 0.0))
                    confidence_pct = confidence * 100 if confidence <= 1.0 else confidence
                    
                    cert = result.get("forensic_certificate", {})

                    st.markdown("### 📊 Scan Results & Audit Verdict")
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.markdown("**Authenticity Status**")
                        if prediction.lower() in ["real", "authentic"]:
                            st.markdown(f'<div class="real-badge">🟢 AUTHENTIC HUMAN VOICE</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="fake-badge">🔴 SYNTHETIC DEEPFAKE / VISHING</div>', unsafe_allow_html=True)
                            
                    with res_col2:
                        st.markdown("**Detection Confidence**")
                        st.metric(label="", value=f"{confidence_pct:.2f}%")
                        st.progress(min(int(confidence_pct), 100))
                        
                    # Forensic Certificate Panel
                    st.markdown("### 📜 Cryptographic Forensic Certificate")
                    st.json(cert)
                    
                    # Download Forensic Report Button
                    cert_json_str = json.dumps(result, indent=4)
                    st.download_button(
                        label="📥 Download Official Forensic Audit Certificate (JSON)",
                        data=cert_json_str,
                        file_name=f"VoiceShield_Audit_{cert.get('certificate_id', 'Report')}.json",
                        mime="application/json"
                    )
                    
                else:
                    st.error(f"Backend Audit Error: {response.json().get('detail', 'Unknown error.')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Connection Error: Could not reach FastAPI backend on port 8000!")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")