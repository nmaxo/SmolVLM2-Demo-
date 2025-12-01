import os
import io
import base64
import requests
import streamlit as st
from PIL import Image

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="SmolVLM VQA & Captioning",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== SIDEBAR CONFIGURATION =====
st.sidebar.title("⚙️ Settings")
default_backend = os.getenv("BACKEND_URL", "http://localhost:8000")
api_url = st.sidebar.text_input("Backend API URL", value=default_backend)

# ===== INITIALIZE SESSION STATE =====
if 'vqa_session_id' not in st.session_state:
    st.session_state['vqa_session_id'] = None
if 'vqa_caption' not in st.session_state:
    st.session_state['vqa_caption'] = None
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None


# ===== HELPER FUNCTIONS =====
def check_backend_health():
    """Check if backend is healthy"""
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def init_vqa_session(uploaded_file, paste_data):
    """Initialize VQA session with image"""
    try:
        # Check backend health
        if not check_backend_health():
            st.error("❌ Backend is offline. Make sure it's running on " + api_url)
            return False

        # Prepare file
        if uploaded_file is not None:
            files = {
                "image": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }
        elif paste_data:
            data = paste_data.strip()
            if data.startswith("data:"):
                _, b64 = data.split(',', 1)
            else:
                b64 = data
            img_bytes = base64.b64decode(b64)
            files = {"image": ("pasted.png", img_bytes, "image/png")}
        else:
            st.warning("Please upload an image or paste image data first")
            return False

        # Send request to backend
        with st.spinner("Processing image and generating caption..."):
            resp = requests.post(
                f"{api_url}/api/vqa/init",
                files=files,
                timeout=120
            )

        if resp.status_code == 200:
            data = resp.json()
            st.session_state['vqa_session_id'] = data.get('session_id')
            st.session_state['vqa_caption'] = data.get('caption')
            st.success("✅ VQA session created successfully!")
            return True
        else:
            try:
                detail = resp.json().get('detail', resp.text)
            except Exception:
                detail = resp.text
            st.error(f"Init failed: {detail}")
            return False

    except Exception as e:
        st.error(f"Request error: {str(e)}")
        return False


def ask_question(session_id, question):
    """Send question to backend and get answer"""
    try:
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{api_url}/api/vqa/ask",
                data={
                    'session_id': session_id,
                    'question': question
                },
                timeout=60
            )

        if resp.status_code == 200:
            return resp.json().get('answer')
        else:
            try:
                detail = resp.json().get('detail', resp.text)
            except Exception:
                detail = resp.text
            st.error(f"Ask failed: {detail}")
            return None

    except Exception as e:
        st.error(f"Request error: {str(e)}")
        return None


# ===== MAIN UI =====
st.title("🖼️ SmolVLM VQA & Captioning")
st.markdown("Upload an image and ask questions about it!")

# ===== IMAGE INPUT SECTION =====
st.header("1️⃣ Upload Image")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG, BMP, TIFF, WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    )

    paste_data = st.text_area(
        "Or paste image data URL (data:image/...) or base64",
        height=80,
        placeholder="Paste data URL or base64 here..."
    )

    # Show preview
    preview_image = None
    if uploaded_file is not None:
        preview_image = Image.open(uploaded_file)
        st.session_state['uploaded_image'] = uploaded_file
    elif paste_data:
        try:
            data = paste_data.strip()
            if data.startswith("data:"):
                header, b64 = data.split(',', 1)
            else:
                b64 = data
            img_bytes = base64.b64decode(b64)
            preview_image = Image.open(io.BytesIO(img_bytes))
            st.session_state['uploaded_image'] = preview_image
        except Exception:
            st.error("Failed to parse image. Check your data URL or base64.")

    if preview_image is not None:
        st.image(preview_image, caption="Selected Image", use_column_width=True)

with col2:
    st.subheader("Session Status")
    
    if st.button("🔄 Init VQA (create session)", use_container_width=True):
        if init_vqa_session(uploaded_file, paste_data):
            st.rerun()

    if st.session_state.get('vqa_session_id'):
        st.success(f"✅ Session Active\nID: {st.session_state['vqa_session_id'][:8]}...")
    else:
        st.info("❌ No active session")

# ===== VQA SECTION =====
if st.session_state.get('vqa_session_id'):
    st.markdown("---")
    st.header("2️⃣ Interact with Image")

    # Show caption
    if st.session_state.get('vqa_caption'):
        st.subheader("📝 Initial Caption")
        st.info(st.session_state['vqa_caption'])

    # Question input
    st.subheader("❓ Ask Questions")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question about the image",
            placeholder="e.g., What color is the main object?"
        )
    
    with col2:
        st.write("")  # Spacer
        ask_button = st.button("Ask ✨", use_container_width=True)

    if ask_button:
        if not question.strip():
            st.error("Question cannot be empty")
        else:
            answer = ask_question(st.session_state['vqa_session_id'], question)
            if answer:
                st.subheader("💬 Answer")
                st.success(answer)

else:
    st.info("👆 Start by uploading an image and clicking 'Init VQA'")

# ===== SIDEBAR STATUS =====
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Backend Status")

if check_backend_health():
    st.sidebar.success("✅ Healthy")
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=5)
        health_data = health_resp.json()
        st.sidebar.write(f"Model: {health_data.get('model', 'Unknown')}")
        st.sidebar.write(f"Device: {health_data.get('device', 'Unknown')}")
    except Exception:
        pass
else:
    st.sidebar.error("❌ Offline")
    st.sidebar.warning(f"Backend URL: {api_url}")

# ===== HELP SECTION =====
with st.expander("📖 Help & Guide"):
    st.markdown("""
    ### How to Use
    
    1. **Upload an Image**
       - Click the uploader to select an image file (JPG, PNG, etc)
       - Or paste a data URL / base64 string
    
    2. **Initialize VQA Session**
       - Click "Init VQA" button
       - The system will generate an initial caption for your image
       - This creates a session that keeps your image in memory
    
    3. **Ask Questions**
       - Type any question about your image
       - Click "Ask" to get an answer
       - You can ask unlimited questions without re-uploading
    
    ### Technical Details
    
    - **Model**: SmolVLM2 (256M parameters)
    - **First run**: 60-120 seconds (model download on first use)
    - **Subsequent runs**: 10-30 seconds per request
    - **Session timeout**: 1 hour of inactivity
    - **Device**: CPU (or GPU if available)
    
    ### Docker Setup
    
    - **Frontend URL**: http://localhost:8501
    - **Backend URL**: http://localhost:8000 (local) or http://backend:8000 (Docker)
    - **Backend health**: http://localhost:8000/health
    
    ### Example Questions
    
    - "What objects are in this image?"
    - "What color is the main subject?"
    - "How many people are there?"
    - "Describe the background"
    - "What is the weather like?"
    """)

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "SmolVLM VQA & Captioning | Powered by Hugging Face</p>",
    unsafe_allow_html=True
)