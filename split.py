import os
import shutil
import random
from pathlib import Path

def create_train_test_split(source_dir, train_dir, test_dir, test_ratio=0.2):
    """
    Manually split dataset ensuring every class has test samples
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for person_name in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        
        # Get all images
        images = [f for f in os.listdir(person_path) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = max(1, int(len(images) * test_ratio))  # At least 1 test image
        
        test_images = images[:split_idx]
        train_images = images[split_idx:]
        
        # Create directories
        train_person_dir = os.path.join(train_dir, person_name)
        test_person_dir = os.path.join(test_dir, person_name)
        os.makedirs(train_person_dir, exist_ok=True)
        os.makedirs(test_person_dir, exist_ok=True)
        
        # Copy files
        for img in train_images:
            shutil.copy2(
                os.path.join(person_path, img),
                os.path.join(train_person_dir, img)
            )
        
        for img in test_images:
            shutil.copy2(
                os.path.join(person_path, img),
                os.path.join(test_person_dir, img)
            )
        
        print(f"{person_name}: {len(train_images)} train, {len(test_images)} test")

# Run this BEFORE training
create_train_test_split(
    "cropped_dataset",
    "dataset_train",
    "dataset_test",
    test_ratio=0.2
)