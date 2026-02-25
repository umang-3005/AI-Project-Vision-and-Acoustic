import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time
import os

# --- CONFIGURATION ---
# Update this path if needed
MODEL_PATH = "rafdb_resnet50_6classes_weighted.pth"

# Verify path
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Check path.")

CAMERA_ID = 0  # chnage according to the webcam ID
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LOAD MODEL ---
print(f"Loading Model on {DEVICE}...")
model = models.resnet50(weights=None)
# 6 Classes (Surprise, Fear, Happy, Sad, Angry, Neutral)
model.fc = nn.Linear(in_features=2048, out_features=6) 

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("[SUCCESS] Model loaded successfully!")  
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}") 
    exit()

model = model.to(DEVICE)
model.eval()

# --- FAST FACE DETECTOR ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- TRANSFORMS ---
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Label Mapping (Must match training!)
EMOTION_LABELS = {
    0: 'Surprise', 
    1: 'Fear', 
    2: 'Happy', 
    3: 'Sad', 
    4: 'Angry', 
    5: 'Neutral'
}

# --- OPTIMIZED LOOP WITH SENSITIVITY HACK ---
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("q Error: Could not open webcam. Try changing CAMERA_ID to 0.")
    exit()

frame_count = 0
SKIP_FRAMES = 3  # Predict every 3rd frame to reduce lag
last_emotions = [] # Cache for skipping

# Replace Unicode green circle with plain text alternative
print("[READY] Starting Inference. Press 'q' to quit.")  # Plain text alternative

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    current_frame_predictions = []

    for i, (x, y, w, h) in enumerate(faces):
        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # LOGIC: Run model only on specific frames OR if new faces appear
        if frame_count % SKIP_FRAMES == 0 or i >= len(last_emotions):
            
            # Crop & Preprocess
            face_roi = frame[y:y+h, x:x+w]
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_face)
            
            try:
                tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    
                    # MAPPING: 
                    # 0:Surprise, 1:Fear, 2:Happy, 3:Sad, 4:Angry, 5:Neutral
                    
                    if probs[1] > 0.15:
                        pred_idx = 1
                        
                    elif probs[0] > 0.40 and probs[1] > 0.10:
                        pred_idx = 1
                        
                    # 3. ANGRY 
                    elif probs[4] > 0.25:
                        pred_idx = 4
                        
                    # 4. SAD 
                    elif probs[3] > 0.35:
                        pred_idx = 3
                        
                    # 5. HAPPY 
                    elif probs[2] > 0.50:
                        pred_idx = 2
                        
                    # 6. SURPRISE 
                    elif probs[0] > 0.50:
                        pred_idx = 0
                        
                    # 7. NEUTRAL
                    else:
                        pred_idx = 5
                    
                    label = EMOTION_LABELS[pred_idx]
                    score = probs[pred_idx].item()
                    
                    current_frame_predictions.append((label, score))
            except Exception as e:
                current_frame_predictions.append(("Error", 0.0))
        else:
            # Use cached prediction for smoothness
            if i < len(last_emotions):
                current_frame_predictions.append(last_emotions[i])
            else:
                current_frame_predictions.append(("...", 0.0))

        # Display Text
        if i < len(current_frame_predictions):
            txt, conf = current_frame_predictions[i]
            
            # Color Logic: Red for negative, Green for positive
            if txt in ['Angry', 'Fear', 'Sad']:
                color = (0, 0, 255) # Red
            else:
                color = (0, 255, 0) # Green

            # Add background for readability
            cv2.putText(frame, f"{txt} {int(conf*100)}%", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Update cache
    if frame_count % SKIP_FRAMES == 0:
        last_emotions = current_frame_predictions

    cv2.imshow('Optimized Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()