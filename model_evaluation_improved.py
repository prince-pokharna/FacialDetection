# model_evaluation_improved.py
import tensorflow as tf
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

print("="*60)
print("Face Recognition Model Evaluation")
print("="*60 + "\n")

# Configuration
IMG_SIZE = 160  # Must match training size
BATCH_SIZE = 16
MODEL_PATH = "best_model_improved.keras"  # Use available model
LABELS_PATH = "labels.json"
TEST_DIR = "dataset_test"

# Load labels
print(f"Loading labels from {LABELS_PATH}...")
with open(LABELS_PATH) as f:
    labels = json.load(f)

num_classes = len(labels)
print(f"Number of classes: {num_classes}")
print(f"Classes: {labels}\n")

# Load model
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[OK] Model loaded successfully!\n")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    print("\nTrying alternative model paths...")
    # Try alternative paths
    alternative_paths = [
        "face_recognition_model_final.keras",
        "best_model_improved_finetuned.keras",
        "best_model_improved.keras",
        "best_model_finetuned.keras",
        "best_model.keras",
    ]
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            print(f"Found: {alt_path}")
            model = tf.keras.models.load_model(alt_path)
            MODEL_PATH = alt_path
            print(f"[OK] Model loaded from {alt_path}\n")
            break
    else:
        raise FileNotFoundError("No model file found!")

# Model summary
print("Model Architecture Summary:")
print(f"Total parameters: {model.count_params():,}")
print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print("\n")

# Load test dataset
print(f"Loading test data from {TEST_DIR}...")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='categorical'
)

# Get dataset size
test_size = tf.data.experimental.cardinality(test_ds).numpy()
print(f"[OK] Test batches: {test_size}")
print(f"  Approximate test images: {test_size * BATCH_SIZE}\n")

# Cache and prefetch for speed
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

print("="*60)
print("Running Evaluation...")
print("="*60 + "\n")

# Collect predictions and true labels
y_true = []
y_pred = []
y_pred_probs = []

batch_count = 0
for images, ys in test_ds:
    batch_count += 1
    print(f"Processing batch {batch_count}/{test_size}...", end='\r')
    
    # Predict
    probs = model.predict(images, verbose=0)
    preds = probs.argmax(axis=1)
    
    # Store results
    y_pred.extend(preds.tolist())
    y_pred_probs.extend(probs.tolist())
    
    # Handle categorical labels
    if len(ys.shape) > 1:  # Categorical format
        y_true.extend(ys.numpy().argmax(axis=1).tolist())
    else:  # Integer format
        y_true.extend(ys.numpy().tolist())

print("\n")

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_probs = np.array(y_pred_probs)

# Calculate overall accuracy
overall_accuracy = accuracy_score(y_true, y_pred)

print("="*60)
print("Evaluation Results")
print("="*60 + "\n")

print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n")

unique_classes = sorted(list(set(y_true.tolist() + y_pred.tolist())))
actual_labels = [labels[i] for i in unique_classes]

print("Classification Report:")
print("-"*60)
print(f"Note: {len(unique_classes)} out of {num_classes} classes present in validation set\n")
report = classification_report(y_true, y_pred, labels=unique_classes, target_names=actual_labels, digits=4)
print(report)

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, labels=unique_classes
)

print("\nPer-Class Detailed Metrics:")
print("-"*60)
print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-"*60)
for i, class_idx in enumerate(unique_classes):
    label_name = labels[class_idx]
    print(f"{label_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")

# Average metrics
print("-"*60)
print(f"{'Macro Average':<20} {precision.mean():<12.4f} {recall.mean():<12.4f} {f1.mean():<12.4f} {support.sum():<10}")
print(f"{'Weighted Average':<20} {np.average(precision, weights=support):<12.4f} {np.average(recall, weights=support):<12.4f} {np.average(f1, weights=support):<12.4f} {support.sum():<10}")
print("="*60 + "\n")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

# Plot Confusion Matrix
plt.figure(figsize=(max(12, len(actual_labels) * 0.6), max(10, len(actual_labels) * 0.5)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=actual_labels, yticklabels=actual_labels,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - Accuracy: {overall_accuracy:.2%}', fontsize=14, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[OK] Confusion matrix saved as 'confusion_matrix.png'\n")
plt.close()

# Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(max(12, len(actual_labels) * 0.6), max(10, len(actual_labels) * 0.5)))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=actual_labels, yticklabels=actual_labels,
            cbar_kws={'label': 'Percentage'})
plt.title(f'Normalized Confusion Matrix - Accuracy: {overall_accuracy:.2%}', fontsize=14, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print("[OK] Normalized confusion matrix saved as 'confusion_matrix_normalized.png'\n")
plt.close()

# Top-K Accuracy
def top_k_accuracy(y_true, y_pred_probs, k=5):
    """Calculate top-k accuracy"""
    k = min(k, num_classes)
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = sum([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return correct / len(y_true)

print("Top-K Accuracy:")
print("-"*60)
for k in [1, 3, 5]:
    if k <= num_classes:
        top_k_acc = top_k_accuracy(y_true, y_pred_probs, k)
        print(f"Top-{k} Accuracy: {top_k_acc:.4f} ({top_k_acc*100:.2f}%)")
print("\n")

# Find best and worst performing classes
class_accuracies = []
for i in range(num_classes):
    mask = y_true == i
    if mask.sum() > 0:
        class_acc = (y_pred[mask] == i).sum() / mask.sum()
        class_accuracies.append((labels[i], class_acc, mask.sum()))

class_accuracies.sort(key=lambda x: x[1], reverse=True)

print("Best Performing Classes (Top 5):")
print("-"*60)
print(f"{'Class':<20} {'Accuracy':<15} {'Samples':<10}")
print("-"*60)
for name, acc, count in class_accuracies[:5]:
    print(f"{name:<20} {acc:.4f} ({acc*100:.2f}%) {count:<10}")

print("\nWorst Performing Classes (Bottom 5):")
print("-"*60)
print(f"{'Class':<20} {'Accuracy':<15} {'Samples':<10}")
print("-"*60)
for name, acc, count in class_accuracies[-5:]:
    print(f"{name:<20} {acc:.4f} ({acc*100:.2f}%) {count:<10}")

print("\n" + "="*60)
print("Evaluation Complete!")
print("="*60)

# Save detailed results
results = {
    "model_path": MODEL_PATH,
    "overall_accuracy": float(overall_accuracy),
    "num_classes": num_classes,
    "total_samples": len(y_true),
    "top_3_accuracy": float(top_k_accuracy(y_true, y_pred_probs, 3)),
    "top_5_accuracy": float(top_k_accuracy(y_true, y_pred_probs, 5)),
    "classification_report": classification_report(y_true, y_pred, labels=unique_classes, target_names=actual_labels, output_dict=True),
    "per_class_accuracy": {name: float(acc) for name, acc, _ in class_accuracies}
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n[OK] Detailed results saved to 'evaluation_results.json'")
print("\nGenerated files:")
print("  - confusion_matrix.png")
print("  - confusion_matrix_normalized.png")
print("  - evaluation_results.json")

