import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import os

# 1. CONFIGURATION & ASSETS
# PATHS
MODEL_PATH = "rafdb_resnet50_6classes_weighted.pth"

# GAME SETTINGS
ROUNDS = ['Happy', 'Sad', 'Surprise', 'Angry', 'Fear', 'Neutral']
ROUND_DURATION = 5  # Seconds players have to hold the face
COUNTDOWN_DURATION = 3 # Seconds before round starts
RESULT_DURATION = 3 # Seconds to show round winner

# EMOTION LABELS (Must match your model's training order!)
# 0:Surprise, 1:Fear, 2:Happy, 3:Sad, 4:Angry, 5:Neutral
MODEL_CLASS_MAP = {
    0: 'Surprise', 1: 'Fear', 2: 'Happy', 
    3: 'Sad', 4: 'Angry', 5: 'Neutral'
}

# REVERSE MAP (To find index of target emotion)
TARGET_TO_INDEX = {v: k for k, v in MODEL_CLASS_MAP.items()}

# COLORS (BGR)
COLOR_P1 = (255, 191, 0)   # Deep Sky Blue (Player 1)
COLOR_P2 = (0, 0, 255)     # Red (Player 2)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (50, 50, 50)

# Set the external camera (index 1) as the main camera
CAMERA_ID = 1

# Added a ModernUI class for improved UI rendering
class ModernUI:
    def __init__(self):
        # Try to load a good font, fallback to default
        try:
            self.font_large = ImageFont.truetype("arial.ttf", 60)
            self.font_med = ImageFont.truetype("arial.ttf", 40)
            self.font_small = ImageFont.truetype("arial.ttf", 25)
            self.emoji_font = ImageFont.truetype("seguiemj.ttf", 60) # Windows Emoji Font
        except:
            print(" Custom font not found. Using default PIL font (might be small).")
            self.font_large = ImageFont.load_default()
            self.font_med = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.emoji_font = ImageFont.load_default()

    def draw_glass_panel(self, img, x, y, w, h, color=(0, 0, 0), alpha=0.6):
        """Draws a semi-transparent rounded box"""
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), 2)

    def draw_pil_text(self, img, text, x, y, font, color=(255,255,255), center=True):
        """Draws text using PIL for smoothness"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            w, h = right - left, bottom - top
        except:
            w, h = draw.textsize(text, font=font) # Old Pillow fallback

        if center:
            x -= w // 2
        draw.text((x+2, y+2), text, font=font, fill=(0,0,0))
        draw.text((x, y), text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_progress_bar(self, img, x, y, w, h, progress, color):
        """Draws a filling bar"""
        cv2.rectangle(img, (x, y), (x+w, y+h), (50, 50, 50), -1) # Background
        fill_w = int(w * progress)
        if fill_w > 0:
            cv2.rectangle(img, (x, y), (x+fill_w, y+h), color, -1) # Fill

# 2. SYSTEM SETUP (Model & Camera)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initializing Game Engine on {device}...")

# Load Model
model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, 6) 

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        # Replace Unicode checkmark and cross emojis with plain text
        print("AI Referee Loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
else:
    print(f" Model not found at {MODEL_PATH}")
    exit()

model = model.to(device)
model.eval()

# Transforms
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. GAME ENGINE CLASS

class EmotionGame:
    def __init__(self):
        self.state = "WAITING" # WAITING, COUNTDOWN, PLAYING, SCORE, GAMEOVER
        self.round_idx = 0
        self.scores = [0, 0] # [Player 1, Player 2]
        
        # Timers
        self.timer_start = 0
        self.round_scores_p1 = [] # To average scores during the round
        self.round_scores_p2 = []
        
        self.winner_text = ""
        
    def start_round(self):
        self.state = "COUNTDOWN"
        self.timer_start = time.time()
        self.round_scores_p1 = []
        self.round_scores_p2 = []
        
    def update(self, p1_face, p2_face):
        current_time = time.time()
        target_emotion = ROUNDS[self.round_idx] if self.round_idx < len(ROUNDS) else "DONE"
        
        # --- LOGIC ---
        if self.state == "WAITING":
            if p1_face is not None and p2_face is not None:
                # Both players ready? Start game!
                self.start_round()

        elif self.state == "COUNTDOWN":
            elapsed = current_time - self.timer_start
            if elapsed > COUNTDOWN_DURATION:
                self.state = "PLAYING"
                self.timer_start = time.time()

        elif self.state == "PLAYING":
            elapsed = current_time - self.timer_start
            
            # Analyze P1
            score1 = self.get_confidence(p1_face, target_emotion)
            self.round_scores_p1.append(score1)
            
            # Analyze P2
            score2 = self.get_confidence(p2_face, target_emotion)
            self.round_scores_p2.append(score2)

            if elapsed > ROUND_DURATION:
                self.calculate_round_winner()
                self.state = "SCORE"
                self.timer_start = time.time()

        elif self.state == "SCORE":
            elapsed = current_time - self.timer_start
            if elapsed > RESULT_DURATION:
                self.round_idx += 1
                if self.round_idx >= len(ROUNDS):
                    self.state = "GAMEOVER"
                else:
                    self.start_round()

    def get_confidence(self, face_img, target_emotion):
        """Run AI model on face and return confidence for target emotion"""
        if face_img is None: return 0.0
        
        try:
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            tensor = val_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                
                # Get index of the target emotion
                target_idx = TARGET_TO_INDEX[target_emotion]
                return probs[target_idx].item()
        except:
            return 0.0

    def calculate_round_winner(self):
        # Average the scores over the round duration for stability
        avg_p1 = np.mean(self.round_scores_p1) if self.round_scores_p1 else 0
        avg_p2 = np.mean(self.round_scores_p2) if self.round_scores_p2 else 0
        
        if avg_p1 > avg_p2:
            self.scores[0] += 1
            self.winner_text = "Player 1 Wins!"
        elif avg_p2 > avg_p1:
            self.scores[1] += 1
            self.winner_text = "Player 2 Wins!"
        else:
            self.winner_text = "Draw!"
            
        print(f"Round Result: P1({avg_p1:.2f}) vs P2({avg_p2:.2f}) -> {self.winner_text}")

# 4. DRAWING HELPER FUNCTIONS

def draw_text_centered(img, text, y, font_scale=1.0, color=(255,255,255), thickness=2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = (img.shape[1] - w) // 2
    # Outline
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+2)
    # Text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_hud(frame, game, ui):
    h, w, _ = frame.shape
    
    # Top Bar
    ui.draw_glass_panel(frame, 0, 0, w, 80, COLOR_BG, 0.8)
    
    # Scores
    cv2.putText(frame, f"P1: {game.scores[0]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_P1, 3)
    
    text_size = cv2.getTextSize(f"P2: {game.scores[1]}", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    cv2.putText(frame, f"P2: {game.scores[1]}", (w - 50 - text_size[0], 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_P2, 3)

    # Center Status
    if game.state == "WAITING":
        draw_text_centered(frame, "WAITING FOR 2 PLAYERS...", 50, 1.0, (200, 200, 200))
    
    elif game.state == "COUNTDOWN":
        target = ROUNDS[game.round_idx]
        countdown = int(COUNTDOWN_DURATION - (time.time() - game.timer_start)) + 1
        draw_text_centered(frame, f"Next: {target}", 40, 0.8)
        draw_text_centered(frame, str(countdown), h//2, 4.0, (0, 255, 255), 5)
        
    elif game.state == "PLAYING":
        target = ROUNDS[game.round_idx]
        timer = int(ROUND_DURATION - (time.time() - game.timer_start)) + 1
        draw_text_centered(frame, f"MIMIC: {target.upper()}!", 50, 1.2, (0, 255, 0))
        draw_text_centered(frame, str(timer), 150, 2.0, (0, 255, 255), 3)

    elif game.state == "SCORE":
        draw_text_centered(frame, game.winner_text, h//2, 2.0, (0, 255, 255), 4)
        
    elif game.state == "GAMEOVER":
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1) # Black screen
        if game.scores[0] > game.scores[1]: msg = "PLAYER 1 WINS!"
        elif game.scores[1] > game.scores[0]: msg = "PLAYER 2 WINS!"
        else: msg = "IT'S A DRAW!"
        
        draw_text_centered(frame, "GAME OVER", h//2 - 50, 2.0, (255, 255, 255))
        draw_text_centered(frame, msg, h//2 + 50, 1.5, (0, 255, 0))

# 5. MAIN LOOP

def run_game():
    cap = cv2.VideoCapture(CAMERA_ID) # Change to 1 for external cam if needed
    
    game = EmotionGame()
    ui = ModernUI()
    
    print("Game Started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect Faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
        p1_face = None
        p2_face = None
        
        # If we found at least 2 faces, sort them (Left=P1, Right=P2)
        if len(faces) >= 2:
            # Sort by X coordinate
            faces = sorted(faces, key=lambda b: b[0])
            
            # Get P1 (Leftmost)
            x1, y1, w1, h1 = faces[0]
            p1_face = frame[y1:y1+h1, x1:x1+w1]
            cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), COLOR_P1, 3)
            cv2.putText(frame, "P1", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_P1, 2)
            
            # Get P2 (Rightmost)
            x2, y2, w2, h2 = faces[-1] # Take the last one in sorted list (Rightmost)
            p2_face = frame[y2:y2+h2, x2:x2+w2]
            cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), COLOR_P2, 3)
            cv2.putText(frame, "P2", (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_P2, 2)
        
        # Update Game Logic
        game.update(p1_face, p2_face)
        
        # Draw Interface
        draw_hud(frame, game, ui)
        
        cv2.imshow("Facial Emotion Battle", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()
