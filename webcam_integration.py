# webcam_attendance_improved.py
import cv2
import json
import sqlite3
import time
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime, date

print("="*60)
print("Face Recognition Attendance System")
print("="*60 + "\n")

# ==================== CONFIGURATION ====================
MODEL_PATH = "face_recognition_model_improved_final.keras"  # Improved model path
LABELS_PATH = "labels.json"
IMG_SIZE = 160  # Updated to match training
CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence to mark attendance (70%)
SMOOTHING_FRAMES = 7  # Number of frames to smooth predictions
COOLDOWN_SECONDS = 30  # Seconds before same person can be marked again

# ==================== LOAD MODEL & LABELS ====================
print("Loading model and labels...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except:
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    print("Trying alternative paths...")
    alternative_paths = ["face_recognition_model_final.keras", "best_model_improved_finetuned.keras", "best_model_improved.keras", "best_model_finetuned.keras", "best_model.keras"]
    for alt_path in alternative_paths:
        try:
            model = tf.keras.models.load_model(alt_path)
            MODEL_PATH = alt_path
            print(f"[OK] Model loaded from {alt_path}")
            break
        except:
            continue

with open(LABELS_PATH) as f:
    labels = json.load(f)

print(f"[OK] Loaded {len(labels)} classes")
print(f"  Classes: {labels[:5]}..." if len(labels) > 5 else f"  Classes: {labels}")
print()

# ==================== DATABASE SETUP ====================
print("Setting up attendance database...")
conn = sqlite3.connect("attendance.db", check_same_thread=False)
c = conn.cursor()

# Create attendance table
c.execute("""CREATE TABLE IF NOT EXISTS attendance (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             person_name TEXT,
             confidence REAL,
             timestamp TEXT,
             date TEXT
)""")
conn.commit()
print("[OK] Database ready\n")

# ==================== FACE DETECTION SETUP ====================
print("Initializing face detector...")
# Using Haar Cascade (simple and fast)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Alternative: Using DNN face detector (more accurate but slower)
USE_DNN_DETECTOR = False  # Set to True for better face detection

if USE_DNN_DETECTOR:
    print("Loading DNN face detector...")
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    try:
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        print("[OK] DNN detector loaded (more accurate)")
    except:
        print("[ERROR] DNN detector files not found, using Haar Cascade")
        USE_DNN_DETECTOR = False
else:
    print("[OK] Using Haar Cascade detector (fast)")

print()

# ==================== WEBCAM SETUP ====================
print("Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Error: Could not open webcam")
    exit()

# Set camera properties for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[OK] Webcam started")
print()

# ==================== TRACKING VARIABLES ====================
# Store recent predictions for smoothing
prediction_history = {}  # face_id -> deque of (name, confidence)
last_attendance_time = {}  # name -> timestamp

# FPS calculation
fps_start_time = time.time()
fps_counter = 0
current_fps = 0

# ==================== HELPER FUNCTIONS ====================

def detect_faces_haar(frame):
    """Detect faces using Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def detect_faces_dnn(frame):
    """Detect faces using DNN (more accurate)"""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faces.append([x1, y1, x2-x1, y2-y1])
    
    return np.array(faces)

def preprocess_face(face_img):
    """Preprocess face image for model"""
    face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_array = face_resized.astype("float32") / 255.0
    face_batch = np.expand_dims(face_array, axis=0)
    return face_batch

def get_smoothed_prediction(face_id, name, confidence):
    """Smooth predictions over multiple frames"""
    if face_id not in prediction_history:
        prediction_history[face_id] = deque(maxlen=SMOOTHING_FRAMES)
    
    prediction_history[face_id].append((name, confidence))
    
    # Get most common prediction
    names = [pred[0] for pred in prediction_history[face_id]]
    confidences = [pred[1] for pred in prediction_history[face_id]]
    
    # Most common name
    most_common_name = max(set(names), key=names.count)
    
    # Average confidence for that name
    avg_confidence = np.mean([conf for n, conf in prediction_history[face_id] if n == most_common_name])
    
    return most_common_name, avg_confidence

def can_mark_attendance(name):
    """Check if enough time has passed since last attendance"""
    if name not in last_attendance_time:
        return True
    
    time_elapsed = time.time() - last_attendance_time[name]
    return time_elapsed > COOLDOWN_SECONDS

def mark_attendance(name, confidence):
    """Mark attendance in database"""
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    date_str = current_time.strftime("%Y-%m-%d")
    
    # Check if already marked today
    c.execute("SELECT * FROM attendance WHERE person_name=? AND date=?", (name, date_str))
    if c.fetchone():
        return False, "Already marked today"
    
    # Check cooldown
    if not can_mark_attendance(name):
        return False, "Cooldown period"
    
    # Insert attendance record
    c.execute("INSERT INTO attendance (person_name, confidence, timestamp, date) VALUES (?, ?, ?, ?)",
              (name, confidence, timestamp, date_str))
    conn.commit()
    
    last_attendance_time[name] = time.time()
    
    print(f"[OK] ATTENDANCE MARKED: {name} ({confidence:.2%}) at {timestamp}")
    return True, "Success"

def draw_ui(frame, faces_data):
    """Draw UI elements on frame"""
    h, w = frame.shape[:2]
    
    # Draw semi-transparent overlay at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw title and info
    cv2.putText(frame, "Face Recognition Attendance", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {current_fps:.1f} | Press 'Q' to quit", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw legend
    legend_y = h - 60
    cv2.putText(frame, f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}", (20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Detected Faces: {len(faces_data)}", (20, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# ==================== MAIN LOOP ====================
print("="*60)
print("System Running - Press 'Q' to quit")
print("="*60 + "\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Error reading frame")
            break
        
        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        if USE_DNN_DETECTOR:
            faces = detect_faces_dnn(frame)
        else:
            faces = detect_faces_haar(frame)
        
        faces_data = []
        
        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            face_id = f"face_{i}"
            
            # Add padding around face
            pad = int(0.2 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            
            # Extract and preprocess face
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                continue
            
            face_batch = preprocess_face(face_img)
            
            # Predict
            predictions = model.predict(face_batch, verbose=0)[0]
            confidence = float(np.max(predictions))
            predicted_idx = int(np.argmax(predictions))
            predicted_name = labels[predicted_idx]
            
            # Smooth prediction
            smooth_name, smooth_confidence = get_smoothed_prediction(
                face_id, predicted_name, confidence
            )
            
            # Determine color based on confidence
            if smooth_confidence >= CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)  # Green - Good
                status = "RECOGNIZED"
            elif smooth_confidence >= 0.5:
                color = (0, 165, 255)  # Orange - Uncertain
                status = "UNCERTAIN"
            else:
                color = (0, 0, 255)  # Red - Low confidence
                status = "UNKNOWN"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label_text = f"{smooth_name} ({smooth_confidence:.0%})"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 20), (x1 + text_w + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label_text, (x1 + 5, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw status
            cv2.putText(frame, status, (x1, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Mark attendance if confident enough
            if smooth_confidence >= CONFIDENCE_THRESHOLD:
                marked, message = mark_attendance(smooth_name, smooth_confidence)
                if marked:
                    # Draw success indicator
                    cv2.putText(frame, "ATTENDANCE MARKED!", (x1, y2 + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            faces_data.append({
                'name': smooth_name,
                'confidence': smooth_confidence,
                'box': (x1, y1, x2, y2)
            })
        
        # Draw UI
        frame = draw_ui(frame, faces_data)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 30:
            fps_end_time = time.time()
            current_fps = fps_counter / (fps_end_time - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0
        
        # Display frame
        cv2.imshow("Face Recognition Attendance System", frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            break

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

finally:
    # Cleanup
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    print("[OK] Cleanup complete")
    print("\nThank you for using Face Recognition Attendance System!")