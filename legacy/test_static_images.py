# test_static_images.py - Test model on static images for presentation
import cv2
import numpy as np
import tensorflow as tf
import json
import os
import random
from datetime import datetime

print("="*60)
print("Static Image Testing - For Presentation")
print("="*60 + "\n")

# Configuration
MODEL_PATH = "face_recognition_model_final.keras"
LABELS_PATH = "labels.json"
IMG_SIZE = 160
TEST_DIR = "dataset_test"
OUTPUT_DIR = "test_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[OK] Model loaded: {MODEL_PATH}\n")
except:
    print(f"[ERROR] Could not load {MODEL_PATH}")
    alternative_models = [
        "best_model_improved_finetuned.keras",
        "face_recognition_model_improved_final.keras",
        "best_model_improved.keras",
        "best_model.keras"
    ]
    for alt in alternative_models:
        if os.path.exists(alt):
            try:
                model = tf.keras.models.load_model(alt)
                MODEL_PATH = alt
                print(f"[OK] Model loaded: {alt}\n")
                break
            except:
                continue

# Load labels
with open(LABELS_PATH) as f:
    labels = json.load(f)

print(f"Model: {MODEL_PATH}")
print(f"Number of classes: {len(labels)}")
print()

# Function to preprocess image
def preprocess_image(img_path):
    """Load and preprocess image for model"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = img_resized.astype("float32") / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img, img_batch

# Function to predict and visualize
def predict_and_visualize(img_path, true_label, output_name):
    """Make prediction and create visualization"""
    # Load and preprocess
    original_img, img_batch = preprocess_image(img_path)
    if original_img is None:
        print(f"[ERROR] Could not load {img_path}")
        return
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)[0]
    top_5_indices = np.argsort(predictions)[-5:][::-1]
    top_5_probs = predictions[top_5_indices]
    
    # Get predicted label
    predicted_idx = top_5_indices[0]
    predicted_label = labels[predicted_idx]
    confidence = top_5_probs[0]
    
    # Check if correct
    is_correct = (predicted_label == true_label)
    
    # Create visualization
    h, w = original_img.shape[:2]
    
    # Create larger canvas for text
    canvas_h = h + 300
    canvas = np.ones((canvas_h, w, 3), dtype=np.uint8) * 255
    canvas[:h, :] = original_img
    
    # Add title
    title = "Face Recognition Test - Model Prediction"
    cv2.putText(canvas, title, (10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add true label
    true_text = f"True Label: {true_label}"
    cv2.putText(canvas, true_text, (10, h + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
    
    # Add predicted label
    pred_text = f"Predicted: {predicted_label} ({confidence:.2%})"
    color = (0, 200, 0) if is_correct else (0, 0, 200)
    cv2.putText(canvas, pred_text, (10, h + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add result status
    status = "CORRECT!" if is_correct else "INCORRECT"
    status_color = (0, 200, 0) if is_correct else (0, 0, 200)
    cv2.putText(canvas, f"Result: {status}", (10, h + 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Add top-5 predictions
    cv2.putText(canvas, "Top 5 Predictions:", (10, h + 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    y_pos = h + 195
    for i, (idx, prob) in enumerate(zip(top_5_indices, top_5_probs)):
        label_name = labels[idx]
        bar_width = int(prob * 300)
        
        # Draw probability bar
        cv2.rectangle(canvas, (10, y_pos - 10), (10 + bar_width, y_pos + 5),
                     (100, 200, 100), -1)
        
        # Add text
        text = f"{i+1}. {label_name}: {prob:.2%}"
        text_color = (0, 100, 0) if label_name == true_label else (0, 0, 0)
        cv2.putText(canvas, text, (320, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        y_pos += 20
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(canvas, timestamp, (10, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Add model name
    cv2.putText(canvas, f"Model: {MODEL_PATH}", (w - 350, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Save result
    output_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Print result
    print(f"Image: {os.path.basename(img_path)}")
    print(f"    True Label: {true_label}")
    print(f"    Predicted: {predicted_label} ({confidence:.2%})")
    print(f"    Status: {'[OK] CORRECT' if is_correct else '[X] INCORRECT'}")
    print(f"    Saved: {output_path}\n")
    
    return is_correct, confidence

# Select random images from test set
print("Selecting random test images...\n")

# Get all person folders
person_folders = [f for f in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, f))]

# Select 3 random people
selected_people = random.sample(person_folders, min(3, len(person_folders)))

print(f"Testing on {len(selected_people)} random people from test set:")
for person in selected_people:
    print(f"  - {person}")
print()

results = []
test_count = 0

for person_id in selected_people:
    person_dir = os.path.join(TEST_DIR, person_id)
    images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
    
    if len(images) > 0:
        # Select one random image from this person
        selected_img = random.choice(images)
        img_path = os.path.join(person_dir, selected_img)
        
        test_count += 1
        output_name = f"test_result_{test_count}_{person_id}.jpg"
        
        is_correct, confidence = predict_and_visualize(img_path, person_id, output_name)
        results.append((person_id, is_correct, confidence))

# Summary
print("="*60)
print("Test Summary")
print("="*60)
print(f"\nTotal images tested: {len(results)}")
correct = sum([1 for _, c, _ in results if c])
print(f"Correct predictions: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
print(f"\nAverage confidence: {np.mean([conf for _, _, conf in results]):.2%}")
print(f"\nResults saved in: {OUTPUT_DIR}/")
print("\nGenerated files:")
for i in range(1, test_count + 1):
    print(f"  - test_result_{i}_*.jpg")

print("\n" + "="*60)
print("Testing Complete!")
print("="*60)
print("\nUse these images for your presentation!")
print("They show live model predictions with confidence scores.")

