Multimodal Emotion Recognition System (Face + Speech)
=====================================================

Project overview
----------------

This project is a **multimodal emotion recognition system** that detects human emotions from:

1) **Face** (webcam / camera feed)
2) **Speech** (microphone / audio)

It includes:

- A **Face Emotion Recognition (FER)** model fine-tuned on RAF-DB
- A **Speech Emotion Recognition (SER)** model fine-tuned from WavLM using multiple speech datasets
- A **multimodal fusion** layer/logic that combines both signals for more reliable emotion prediction
- An optional **gamification feature** where two players act an emotion and the system decides who performed it better

Main functionality of the project
---------------------------------

1) **Real-time face emotion detection**

   - Detects faces from a camera feed and predicts emotion probabilities per detected face.
2) **Real-time speech emotion detection**

   - Captures live microphone audio, extracts features, and predicts emotion probabilities.
3) **Multimodal emotion prediction (face + speech together)**

   - Combines the FER and SER outputs into a single, more robust emotion estimate.
   - Fusion reduces failures from single-modality issues (bad lighting vs noisy audio).
4) Optional gamification module

   - Two players perform an emotion at the same time that given by the system.
   - The system predicts emotions for both players and compares who matches the target emotion better.
   - This is a feature built on top of the multimodal system

Shared emotion classes (alignment across both models)
-----------------------------------------------------

Both FER and SER are aligned to the same **6 emotion labels** so they can be fused cleanly:

  ["angry", "fear", "happy", "neutral", "sad", "surprise"]

“Disgust” is intentionally excluded to keep both modalities consistent.

System architecture (high-level)
--------------------------------

A) Face branch (FER)
   camera frames → face detection → face crop → FER model → emotion probabilities

B) Speech branch (SER)
   mic audio → speech segmentation (simple VAD) → SER model → emotion probabilities

C) Fusion (multimodal)
   FER + SER probabilities → combined final emotion

Component 1 — Face Emotion Recognition (FER)
---------------------------------------------

Dataset:

- RAF-DB (Real-world Affective Faces Database), (Link: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)

Label setup:

- Original RAF-DB has 7 emotions (including “disgust”).
- This project removes “disgust” to match the shared 6-class label space.

Model:

- ResNet-50 (pretrained) fine-tuned for 6 classes.(https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)

Training decisions (summary):

- Replace the final fully-connected layer for 6-class output
- Use **weighted cross-entropy** to handle class imbalance
- Apply **oversampling** (Fear, Angry, Sad) to improve minority-class learning
- Use data augmentation
- Two-phase training:
  - Phase 1: train classification head
  - Phase 2: fine-tune deeper layers

Saved model (example):

- rafdb_resnet50_6classes_weighted.pth

Component 2 — Speech Emotion Recognition (SER)
-----------------------------------------------

Datasets used:

- RAVDESS (acted emotional speech), (Link: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- CREMA-D (acted emotional speech), (Link: https://www.kaggle.com/datasets/ejlok1/cremad)
- MELD (dialogue emotional speech), (Link: https://www.kaggle.com/datasets/zaber666/meld-dataset)

Model:

- microsoft/wavlm-base fine-tuned for 6-class audio emotion classification.(https://huggingface.co/microsoft/wavlm-base)

Key training decisions (summary):

- Convert all dataset labels into the shared 6 emotions:
  angry, fear, happy, neutral, sad, surprise
- Drop “disgust” across datasets for label alignment
- Build dataset manifests and enforce **speaker-disjoint** train/val/test splits
  (prevents leakage where the same speaker appears in train and test)
- Use **weighted cross-entropy** for class imbalance
- Select best checkpoint using **macro-F1** (more fair under imbalance)

Optional personalization (small custom recorded dataset):

- Fine-tune only the classifier head (freeze the WavLM encoder)
- Early stopping to reduce overfitting

Saved outputs:

- wavlm_ser_base/   (base fine-tuned model)
- wavlm_ser_model/  (final adapted model used for mic inference)

Multimodal fusion (how the two parts are combined)
--------------------------------------------------

The fusion step combines FER and SER predictions into a single decision. A typical approach is:

  score(emotion) = w_face * P_face(emotion) + w_speech * P_speech(emotion)

Where:

- P_face(emotion)  = probability/confidence from the face model
- P_speech(emotion)= probability/confidence from the speech model
- w_face, w_speech = weights (configurable)

Fusion can be:

- fixed weights (simple and stable)
- dynamic weights (reduce influence of low-quality input: e.g., noisy audio or blurred face)

Project features recap
----------------------

- Real-time facial emotion detection (webcam)
- Real-time speech emotion detection (microphone)
- Multimodal emotion recognition (face + speech fusion)
- Optional gamification feature (two-player emotion challenge)

Repository files, entry points, and outputs
-------------------------------------------

This README explains the *system idea*. The actual runnable demos live in these files:

1) multimodal_system.py  (MAIN demo: Face + Speech together)

   - What it does:
     - Runs **real-time face detection + face emotion prediction**
     - When your face emotion becomes stable (“locked”), it triggers **microphone recording**
     - Runs **speech emotion prediction** and shows both results side-by-side
   - Main function to run:
     - run_system()  (called automatically by `if __name__ == "__main__":`)
   - Important helper functions (high level):
     - audio_worker(): records audio in a background thread and calls speech prediction
     - predict_speech(): preprocesses audio (normalize + resample to 16k) and runs WavLM inference
   - Output after running:
     - Opens an OpenCV window titled **“Multimodal System”**
     - UI states you’ll see:
       - WAITING_FACE: shows face box + “HOLD AN EXPRESSION TO LOCK IN…”
       - LISTENING: screen dims + “SPEAK NOW…” while recording
       - SHOW_RESULT: split screen showing **FACE emotion** vs **VOICE emotion (+confidence)**
     - Console output includes model loading logs and audio recording start/stop messages.
2) run_webcam.py  (Face-only FER realtime demo)

   - What it does:
     - Runs webcam face detection + **face emotion classification** only (no audio).
   - Entry point:
     - This file runs directly as a script (top-level while-loop).
   - Output after running:
     - Opens an OpenCV window titled **“Optimized Emotion Detector”**
     - Draws face boxes + predicted label with confidence (%) on each face.
     - Console shows “[READY] Starting Inference. Press 'q' to quit.”
3) emotion_game.py  (Two-player emotion “battle” game)

   - What it does:
     - Detects **two faces** (left = Player 1, right = Player 2)
     - The game selects a target emotion each round and checks who matches it better.
   - Main function to run:
     - run_game()  (called automatically by `if __name__ == "__main__":`)
   - Important helper functions (high level):
     - EmotionGame.update(): game state machine (WAITING → COUNTDOWN → PLAYING → SCORE → GAMEOVER)
     - EmotionGame.get_confidence(): runs the face model and returns confidence for the target emotion
   - Output after running:
     - Opens an OpenCV window titled **“Facial Emotion Battle”**
     - Shows player boxes, score HUD, countdown timers, and round winner text.
     - Console prints round results like: “Round Result: P1(...) vs P2(...) -> Player X Wins!”
4) FER_training.ipynb  (Training / experimentation notebook)

   - What it does:
     - Notebook used for experiments (installing packages, training/evaluating models, etc.).
   - Output after running:
     - Produces training logs/plots in Jupyter and saves model weights.
5) SER_training.ipynb  (SER training notebook: build datasets + train + save model)

   - What it does:
     - Downloads / prepares speech datasets (RAVDESS, CREMA-D, MELD) and maps them to the shared 6 emotion labels
     - Builds manifest CSVs, checks class imbalance, and trains a **WavLM** audio classifier with a **weighted loss**
     - Optionally adapts (fine-tunes) the final SER model on a small custom dataset (speaker-disjoint split)
   - Main/important functions/classes:
     - build_ravdess_manifest_from_files(), build_crema_manifest_from_files(), build_meld_manifest(): dataset → manifest builders
     - assign_split(): creates speaker-disjoint train/val/test splits
     - AdaptDS: lightweight dataset wrapper (+ augmentation)
     - WeightedTrainer / HeadOnlyTrainer: training logic (weighted loss + head-only fine-tuning)
   - Output after running:
     - Saves manifest CSVs (e.g., `ravdess_manifest.csv`, `crema_manifest.csv`, `meld_manifest.csv`, combined final manifest)
     - Saves the trained SER model as:
       - `wavlm_ser_model/` (HuggingFace model folder for feature extractor/config)
       - `wavlm_ser_model.pth` (PyTorch weights used by the live demos)
6) run_microphone.ipynb  (SER live microphone inference notebook)

   - What it does:
     - Loads the final saved SER model (`wavlm_ser_model/` + `wavlm_ser_model.pth`)
     - Lists microphone devices and runs **real-time streaming prediction** from the mic
   - Main/important functions:
     - resample_to_16k(): converts audio to the WavLM-required 16 kHz
     - rms(): simple loudness measure used for a basic voice-activity trigger
     - predict(): runs the model and prints Top-3 predicted emotions with probabilities
   - Output after running:
     - Prints available input devices (so you can pick the right MIC_INDEX)
     - Console prints predictions like: `PRED: happy=0.72 | neutral=0.18 | surprise=0.06`

Model / asset files required (expected next to the scripts)
-----------------------------------------------------------

These scripts are not useful without the trained weights:

- rafdb_resnet50_6classes_weighted.pth
  - Used by: emotion_game.py, run_webcam.py, multimodal_system.py
- wavlm_ser_model.pth
  - Used by: multimodal_system.py, run_inference.ipynb
  - Generated by: SER_training.ipynb
- wavlm_ser_model/   (folder)
  - Used by: multimodal_system.py, run_inference.ipynb (offline/local HuggingFace feature extractor + config)
  - Generated by: SER_training.ipynb

Quick run commands
------------------

From the project folder:

- Main multimodal system:
  `python multimodal_system.py`
- Face-only webcam demo:
  `python run_webcam.py`
- SER live inference (notebook):
  Open `run_microphone.ipynb` to stream from your microphone and print real-time SER predictions.
- Two-player game:
  `python emotion_game.py`

All OpenCV windows close with **q**.

Hardware config note (important)
--------------------------------

The scripts are currently configured for a specific setup:

- CAMERA_ID = 1 (external webcam)
- MIC_ID = 14 and MIC_NATIVE_SR = 48000 (Fifine microphone)

If your hardware IDs are different, update them at the top of the relevant script.

Limitations
-----------

- Emotion labels are subjective and noisy.
- Acted datasets (RAVDESS/CREMA-D) don’t perfectly match real-life expression.
- Camera conditions (lighting/angle/occlusion) affect face accuracy.
- Noise and microphone quality affect speech accuracy.
- Simple RMS-based VAD can trigger incorrectly in loud environments.

Credits / acknowledgements
--------------------------

- WavLM (Microsoft) pretrained speech model
- RAF-DB dataset for face emotion recognition
- RAVDESS, CREMA-D, MELD datasets for speech emotion recognition
- PyTorch + torchaudio
- Hugging Face Transformers + Datasets
