# preprocess.py
# This script is part of a face recognition data pipeline.
# It detects, crops, and resizes faces from raw images using Haar Cascade, applying preprocessing for training.
# Input: A directory containing identity subfolders with raw images (e.g., 'subset').
# Output: A directory with preprocessed and cropped face images (e.g., 'cropped_dataset') resized to IMG_SIZE.

import cv2
import os
import glob
import sys
from tqdm import tqdm

# Configuration
INPUT_ROOT = "subset"
OUT_ROOT = "cropped_dataset"
IMG_SIZE = 128
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Basic error handling
if not os.path.exists(INPUT_ROOT):
    print(f"Error: Expected input directory '{INPUT_ROOT}' does not exist.")
    print("Please make sure you have run the subset generation (sub.py and subred.py) first.")
    sys.exit(1)

print("="*60)
print("Face Detection and Preprocessing - IMPROVED")
print("="*60 + "\n")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Statistics tracking
total_images = 0
detected_faces = 0
failed_detections = 0

# Create output directory
os.makedirs(OUT_ROOT, exist_ok=True)

print(f"Input directory: {INPUT_ROOT}")
print(f"Output directory: {OUT_ROOT}")
print(f"Target image size: {IMG_SIZE}x{IMG_SIZE}\n")

# Get all person folders
person_folders = [f for f in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, f))]
print(f"Found {len(person_folders)} person folders\n")

# Process each person
for person_idx, person_folder in enumerate(person_folders, 1):
    src = os.path.join(INPUT_ROOT, person_folder)
    dst = os.path.join(OUT_ROOT, person_folder)
    os.makedirs(dst, exist_ok=True)
    
    # Get all images for this person
    img_paths = glob.glob(os.path.join(src, "*.jpg"))
    person_detected = 0
    person_failed = 0
    
    print(f"[{person_idx}/{len(person_folders)}] Processing {person_folder} ({len(img_paths)} images)...", end=' ')
    
    for img_path in img_paths:
        total_images += 1
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            person_failed += 1
            failed_detections += 1
            continue
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with improved parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            person_failed += 1
            failed_detections += 1
            continue
        
        # Take largest face
        x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        
        # Add 20% padding around face
        padding = int(0.2 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        # Extract face with padding
        face = img[y1:y2, x1:x2]
        
        # Resize to target size
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply slight denoising
        face = cv2.fastNlMeansDenoisingColored(face, None, 10, 10, 7, 21)
        
        # Save processed face
        out_path = os.path.join(dst, os.path.basename(img_path))
        cv2.imwrite(out_path, face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        person_detected += 1
        detected_faces += 1
    
    # Print results for this person
    detection_rate = (person_detected / len(img_paths) * 100) if len(img_paths) > 0 else 0
    print(f"Detected: {person_detected}/{len(img_paths)} ({detection_rate:.1f}%)")

# Final statistics
print("\n" + "="*60)
print("Preprocessing Complete!")
print("="*60)
if total_images > 0:
    print(f"\nTotal images processed: {total_images}")
    print(f"Successfully detected: {detected_faces} ({detected_faces/total_images*100:.1f}%)")
    print(f"Failed detections: {failed_detections} ({failed_detections/total_images*100:.1f}%)")
else:
    print("\nNo images found to process.")
print(f"\nOutput saved to: {OUT_ROOT}/")
print("\nNext step: Run 'python split.py' to create train/test split")
