import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image

# --- 1. AI MODEL SETUP ---
# We initialize these once to save cloud memory
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.5
)

# Load the classic Haar Cascade for Face Detection
# This file is built into the opencv-python-headless library
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Safety Interlock", layout="wide")
st.title("⚡ AI 'Dead-Man's Switch' Simulator")
st.markdown("### 3rd Year AI Engineering Project | Prajal Jain")

# --- 3. SESSION STATE (CIRCUIT LOGIC) ---
if 'last_seen' not in st.session_state:
    st.session_state.last_seen = time.time()
if 'circuit_status' not in st.session_state:
    st.session_state.circuit_status = "OPEN"

# --- 4. UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 Electrical Dashboard")
    v_metric = st.empty()
    i_metric = st.empty()
    p_metric = st.empty()
    status_box = st.empty()
    st.info("💡 Tip: Raise your Left hand for Voltage, Right hand for Current.")

with col1:
    # This activates the browser's camera permissions
    img_file = st.camera_input("Operator Presence Verification")

# --- 5. CORE PROCESSING ENGINE ---
voltage, current = 0.0, 0.0

if img_file:
    # Convert Streamlit file to OpenCV format
    img = Image.open(img_file)
    frame = np.array(img)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # STEP A: Face Detection (The Safety Interlock)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_detected = len(faces) > 0
    
    if face_detected:
        st.session_state.last_seen = time.time()
        st.session_state.circuit_status = "CLOSED"
        
        # STEP B: Hand Tracking (Parameter Control)
        results = hands.process(frame)
        voltage, current = 12.0, 4.0 # Base operational values
        
        if results.multi_hand_landmarks:
            for res, info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = info.classification[0].label # 'Left' or 'Right'
                wrist_y = res.landmark[0].y # Height (0.0 to 1.0)
                
                if label == 'Left':
                    # Scale voltage between 12V and 36V based on hand height
                    voltage = round(12.0 + (1.0 - wrist_y) * 24, 1)
                if label == 'Right':
                    # Scale current between 4A and 16A
                    current = round(4.0 + (1.0 - wrist_y) * 12, 1)
    else:
        # STEP C: The 5-Second Safety Buffer
        elapsed = time.time() - st.session_state.last_seen
        if elapsed > 5:
            st.session_state.circuit_status = "OPEN"
        else:
            st.warning(f"⚠️ OPERATOR MISSING! Shutdown in {max(0, int(5.5 - elapsed))}s")

# --- 6. OUTPUT & CALCULATIONS ---
if st.session_state.circuit_status == "CLOSED":
    power = round(voltage * current, 1)
    v_metric.metric("Voltage", f"{voltage} V")
    i_metric.metric("Current", f"{current} A")
    p_metric.metric("Power", f"{power} W")
    status_box.success("✅ CIRCUIT LIVE")
else:
    # Total Shutdown State
    v_metric.metric("Voltage", "0.0 V")
    i_metric.metric("Current", "0.0 A")
    p_metric.metric("Power", "0.0 W")
    status_box.error("🛑 CIRCUIT OPEN (SHUTDOWN)")
