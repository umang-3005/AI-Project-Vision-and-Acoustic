import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import threading
import time
import os
import sounddevice as sd
import torchaudio
import json
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# 1. CONFIGURATION

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DYNAMIC PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_PATH = os.path.join(SCRIPT_DIR, "rafdb_resnet50_6classes_weighted.pth")
SPEECH_MODEL_PATH = os.path.join(SCRIPT_DIR, "wavlm_ser_model.pth")
SPEECH_CONFIG_DIR = os.path.join(SCRIPT_DIR, "wavlm_ser_model")

# HARDWARE SETTINGS (CRITICAL)
CAMERA_ID = 1       # External Webcam
MIC_ID = 16         # <-- your Fifine index that works
MIC_NATIVE_SR = 48000 # The specific rate for Fifine

# LABELS
FACE_LABELS = {0: 'Surprise', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Angry', 5: 'Neutral'}
SPEECH_LABELS_MAP = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

# TIMING
FACE_DETECT_TIME = 4.0   # Seconds to watch face
SPEECH_RECORD_TIME = 5.0 # Seconds to record audio

# SHARED STATE
current_state = "WAITING_FACE"
locked_face_emotion = ""
final_voice_emotion = ""
trigger_audio = False 

# 2. MODEL LOADERS

# FACE MODEL 
print(f"[SYSTEM] Loading Face Model from: {FACE_MODEL_PATH}")
face_model = models.resnet50(weights=None)
face_model.fc = nn.Linear(2048, 6)
try:
    face_model.load_state_dict(torch.load(FACE_MODEL_PATH, map_location=DEVICE))
    face_model.to(DEVICE)
    face_model.eval()
    print("[SUCCESS] Face Model Loaded.")
except Exception as e:
    print(f"[CRITICAL ERROR] Face Model Failed: {e}")
    exit()

# SPEECH MODEL 
print(f"[SYSTEM] Loading Speech Model...")
try:
    # Load Feature Extractor
    if os.path.exists(SPEECH_CONFIG_DIR):
        feature_extractor = AutoFeatureExtractor.from_pretrained(SPEECH_CONFIG_DIR, local_files_only=True)
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")

    # Load Architecture (wavlm-base, 6 labels)
    speech_model = AutoModelForAudioClassification.from_pretrained(
        "microsoft/wavlm-base", 
        num_labels=6
    )

    # Load Weights
    state_dict = torch.load(SPEECH_MODEL_PATH, map_location=DEVICE)
    speech_model.load_state_dict(state_dict)
    speech_model.to(DEVICE)
    speech_model.eval()
    print("[SUCCESS] Speech Model Loaded.")

except Exception as e:
    print(f"[CRITICAL ERROR] Speech Model Failed: {e}")
    exit()

# 3. AUDIO PROCESSING

TARGET_SR = 16000 # WavLM requires 16k

def predict_speech(audio_numpy, mic_sr):
    """
    Replicates SER.ipynb preprocessing exactly.
    """
    # Normalize Loudness (CRITICAL)
    peak = np.max(np.abs(audio_numpy)) + 1e-9
    audio_norm = (audio_numpy / peak).astype(np.float32)
    
    # Convert to Tensor
    audio_tensor = torch.tensor(audio_norm).unsqueeze(0).to(DEVICE) # [1, T]
    
    # Resample to 16k
    if mic_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(mic_sr, TARGET_SR).to(DEVICE)
        audio_16k = resampler(audio_tensor)
    else:
        audio_16k = audio_tensor
    
    # Feature Extraction
    audio_16k_np = audio_16k.squeeze().cpu().numpy()
    
    inputs = feature_extractor(
        [audio_16k_np], 
        sampling_rate=TARGET_SR, 
        padding=True, 
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        logits = speech_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()
        
    return SPEECH_LABELS_MAP[pred_id], confidence

# 4. AUDIO WORKER THREAD
def audio_worker():
    global current_state, final_voice_emotion, trigger_audio
    
    print(f"[AUDIO] Initializing Fifine Mic (Index {MIC_ID}) at {MIC_NATIVE_SR}Hz...")

    while True:
        if not trigger_audio:
            time.sleep(0.1)
            continue
            
        current_state = "LISTENING"
        try:
            print(">>> RECORDING STARTED...")
            # Record at Microphone's NATIVE rate (48000)
            recording = sd.rec(int(SPEECH_RECORD_TIME * MIC_NATIVE_SR), 
                               samplerate=MIC_NATIVE_SR, channels=1, device=MIC_ID, dtype='float32')
            sd.wait()
            print("<<< RECORDING STOPPED.")
            
            # Predict
            audio_flat = recording.flatten()
            emotion, conf = predict_speech(audio_flat, MIC_NATIVE_SR)
            
            final_voice_emotion = f"{emotion} ({int(conf*100)}%)"
            print(f"   [RESULT] {final_voice_emotion}")
            
            current_state = "SHOW_RESULT"
            trigger_audio = False 
            
            # Show result for 7 seconds then reset
            time.sleep(7)
            current_state = "WAITING_FACE"
            
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
            final_voice_emotion = "Error"
            trigger_audio = False
            current_state = "WAITING_FACE"

# 5. VIDEO LOOP
def run_system():
    global current_state, locked_face_emotion, trigger_audio
    
    # Start Audio Thread
    t = threading.Thread(target=audio_worker)
    t.daemon = True
    t.start()
    
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[ERROR] Camera {CAMERA_ID} failed to open.")
        return
    
    # Face Transforms
    face_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Buffer for stability
    face_buffer = []
    STABILITY_REQ = 10 
    
    print("[SYSTEM] Video UI Started.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        
        # STATE 1: DETECT FACE
        if current_state == "WAITING_FACE":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
            
            if len(faces) > 0:
                # Get largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, fw, fh) = faces[0]
                
                # Draw Box
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (255, 255, 0), 2)
                
                # INFERENCE
                try:
                    # Padding logic (from run_webcam.py)
                    pad_x = max(0, x - 10)
                    pad_y = max(0, y - 10)
                    pad_w = min(w, x + fw + 10) - pad_x
                    pad_h = min(h, y + fh + 10) - pad_y
                    
                    face_roi = frame[pad_y:pad_y+pad_h, pad_x:pad_x+pad_w]
                    
                    if face_roi.size > 0:
                        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_face)
                        tensor = face_transform(pil_img).unsqueeze(0).to(DEVICE)
                        
                        with torch.no_grad():
                            probs = torch.softmax(face_model(tensor), dim=1)[0]
                            
                            # SENSITIVITY LOGIC (From run_webcam.py)
                            if probs[1] > 0.15: idx = 1   # Fear (Aggressive)
                            elif probs[4] > 0.25: idx = 4 # Angry
                            elif probs[3] > 0.35: idx = 3 # Sad
                            elif probs[0] > 0.50: idx = 0 # Surprise
                            elif probs[2] > 0.50: idx = 2 # Happy
                            else: idx = 5                 # Neutral
                            
                            curr_lbl = FACE_LABELS[idx]
                            
                            # Stability Check
                            face_buffer.append(curr_lbl)
                            if len(face_buffer) > STABILITY_REQ: face_buffer.pop(0)
                            
                            # Auto-Lock if stable
                            if len(face_buffer) == STABILITY_REQ and len(set(face_buffer)) == 1:
                                locked_face_emotion = curr_lbl
                                current_state = "LOCKED_FACE"
                                trigger_audio = True
                                face_buffer = [] 
                                
                            cv2.putText(frame, f"{curr_lbl}", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                except Exception as e:
                    pass
            else:
                face_buffer = []

            # Instructions
            cv2.rectangle(frame, (0, h-60), (w, h), (0,0,0), -1)
            cv2.putText(frame, "HOLD AN EXPRESSION TO LOCK IN...", (20, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # STATE 2: LISTENING
        elif current_state == "LISTENING":
            # Dim screen
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Display Locked Face
            cv2.putText(frame, "FACE LOCKED:", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, locked_face_emotion, (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            
            # Blink Instruction
            if (time.time() * 2) % 2 > 1:
                cv2.putText(frame, "SPEAK NOW...", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

        # STATE 3: RESULTS
        elif current_state == "SHOW_RESULT":
            # Split Screen Visual
            cv2.rectangle(frame, (0,0), (w//2, h), (50,50,50), -1)
            cv2.rectangle(frame, (w//2,0), (w, h), (30,30,30), -1)
            
            # Left: Face
            cv2.putText(frame, "FACE", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
            cv2.putText(frame, locked_face_emotion, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            
            # Right: Speech
            cv2.putText(frame, "VOICE", (w//2 + 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
            cv2.putText(frame, final_voice_emotion, (w//2 + 50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            
            cv2.putText(frame, "Resetting...", (w//2 - 50, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        cv2.imshow("Multimodal System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()