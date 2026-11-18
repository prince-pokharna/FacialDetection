# model_train_improved.py - Enhanced Training with Advanced Techniques
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
import json, os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight

print("="*60)
print("Face Recognition Model Training - IMPROVED")
print("="*60 + "\n")

# Configuration
IMG_SIZE = 160
BATCH_SIZE = 32  # Increased from 16
EPOCHS_PHASE1 = 60  # Increased
EPOCHS_PHASE2 = 40  # Increased
DATA_DIR = "cropped_dataset"
TRAIN_DIR = "dataset_train"
TEST_DIR = "dataset_test"

# Enhanced Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),  # Increased from 0.15
    layers.RandomZoom(0.2),  # Increased from 0.15
    layers.RandomTranslation(0.15, 0.15),  # Increased from 0.1
    layers.RandomContrast(0.3),  # Increased from 0.2
    layers.RandomBrightness(0.3),  # Increased from 0.2
    # Add new augmentations
    layers.GaussianNoise(0.05),  # Add noise for robustness
], name="data_augmentation")

# Load datasets with validation split
print("Loading dataset...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    seed=42
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    seed=42
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Number of people: {num_classes}")
print(f"Names: {class_names}\n")

# Calculate class weights for handling imbalance
print("Calculating class weights for balanced training...")
# Get all labels
all_labels = []
for _, labels in train_ds:
    all_labels.extend(np.argmax(labels.numpy(), axis=1))

# Calculate class weights
class_weights_array = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
class_weights_dict = dict(enumerate(class_weights_array))
print(f"Class weights calculated (min: {min(class_weights_array):.2f}, max: {max(class_weights_array):.2f})")
print()

# Calculate dataset sizes
train_size = tf.data.experimental.cardinality(train_ds).numpy()
val_size = tf.data.experimental.cardinality(val_ds).numpy()
print(f"Training batches: {train_size}")
print(f"Validation batches: {val_size}")
print(f"Training images: ~{train_size * BATCH_SIZE}")
print(f"Validation images: ~{val_size * BATCH_SIZE}\n")

# Optimize data pipeline with better caching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Build improved model using MobileNetV2 with enhancements
def build_improved_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=num_classes):
    """
    Build an improved model with MobileNetV2 base and enhanced architecture
    """
    from tensorflow.keras.applications import MobileNetV2
    
    # Load MobileNetV2 (reliable and efficient)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build complete model
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation
    x = data_augmentation(inputs)
    
    # Preprocessing for MobileNetV2 (rescaling to -1 to 1)
    x = layers.Rescaling(1./127.5, offset=-1)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Enhanced custom layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)  # Add batch normalization
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer with label smoothing through from_logits=False
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

print("Building improved model with MobileNetV2 + enhancements...\n")
model, base_model = build_improved_model()

# Compile with label smoothing
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # Add label smoothing
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print("Model Summary:")
model.summary()
print(f"\nTotal parameters: {model.count_params():,}")
print()

# Setup enhanced callbacks
os.makedirs("logs", exist_ok=True)

# Cosine decay learning rate schedule
def get_cosine_decay_schedule(initial_learning_rate, epochs, steps_per_epoch):
    decay_steps = epochs * steps_per_epoch
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, alpha=0.0
    )

cbs = [
    callbacks.ModelCheckpoint(
        "best_model_improved.keras",
        save_best_only=True,
        monitor="val_accuracy",
        mode='max',
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,  # Increased patience
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1
    )
]

print("\n" + "="*60)
print("Training Phase 1: Training top layers")
print("="*60 + "\n")

# Train Phase 1 with class weights
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    callbacks=cbs,
    class_weight=class_weights_dict,  # Apply class weights
    verbose=1
)

# Fine-tuning Phase
print("\n" + "="*60)
print("Training Phase 2: Fine-tuning (unfreezing more layers)")
print("="*60 + "\n")

# Unfreeze more layers for fine-tuning
base_model.trainable = True

# Freeze the first 100 layers (or less if model has fewer), train the rest
freeze_until = min(100, len(base_model.layers) - 30)
for i, layer in enumerate(base_model.layers):
    if i < freeze_until:
        layer.trainable = False

print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")
print(f"Frozen layers: {sum([1 for layer in model.layers if not layer.trainable])}")
print()

# Recompile with much lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),  # Very low learning rate
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

# Update checkpoint path
cbs[0] = callbacks.ModelCheckpoint(
    "best_model_improved_finetuned.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode='max',
    verbose=1
)

# Continue training with fine-tuning
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    callbacks=cbs,
    class_weight=class_weights_dict,
    verbose=1
)

# Combine histories
history.history['accuracy'].extend(history_fine.history['accuracy'])
history.history['val_accuracy'].extend(history_fine.history['val_accuracy'])
history.history['loss'].extend(history_fine.history['loss'])
history.history['val_loss'].extend(history_fine.history['val_loss'])

if 'top_3_accuracy' in history_fine.history:
    if 'top_3_accuracy' not in history.history:
        history.history['top_3_accuracy'] = []
        history.history['val_top_3_accuracy'] = []
    history.history['top_3_accuracy'].extend(history_fine.history['top_3_accuracy'])
    history.history['val_top_3_accuracy'].extend(history_fine.history['val_top_3_accuracy'])

# Save final model
model.save("face_recognition_model_improved_final.keras")
with open("labels.json", "w") as f:
    json.dump(class_names, f, indent=2)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# Plot comprehensive results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy plot
axes[0, 0].plot(history.history['accuracy'], label='Training', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss plot
axes[0, 1].plot(history.history['loss'], label='Training', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Top-3 Accuracy (if available)
if 'top_3_accuracy' in history.history:
    axes[1, 0].plot(history.history['top_3_accuracy'], label='Training', linewidth=2)
    axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation', linewidth=2)
    axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-3 Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# Learning rate (if tracked)
axes[1, 1].text(0.5, 0.5, 'Training Statistics\n\n' + 
                f'Final Train Acc: {history.history["accuracy"][-1]:.2%}\n' +
                f'Final Val Acc: {history.history["val_accuracy"][-1]:.2%}\n' +
                f'Best Val Acc: {max(history.history["val_accuracy"]):.2%}\n' +
                f'Total Epochs: {len(history.history["accuracy"])}',
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'training_history_improved.png'")

# Final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
best_val_acc = max(history.history['val_accuracy'])

print(f"\nFinal Training Accuracy: {final_train_acc:.2%}")
print(f"Final Validation Accuracy: {final_val_acc:.2%}")
print(f"Best Validation Accuracy: {best_val_acc:.2%}")
print(f"\nModels saved:")
print(f"  - face_recognition_model_improved_final.keras (final model)")
print(f"  - best_model_improved_finetuned.keras (best checkpoint)")
print(f"  - labels.json (class labels)")
print(f"\nNext step: Run 'python model_evaluation_improved.py' to evaluate!")

