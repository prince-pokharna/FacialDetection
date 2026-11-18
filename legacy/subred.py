# subred.py
# This script is part of a face recognition data pipeline.
# It reduces the number of images per identity folder to a specified limit in-place to save storage/training time.
# Input: A directory containing identity subfolders (e.g., 'subset'), with multiple images per subfolder.
# Output: The same directory with at most images_per_person images per identity subfolder (excess images are deleted).

import os
import random
import sys

# Configuration variables
SUBSET_PATH = r'subset'
IMAGES_PER_PERSON = 150
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# Basic error handling
if not os.path.exists(SUBSET_PATH):
    print(f"Error: Expected subset directory '{SUBSET_PATH}' does not exist.")
    print("Please make sure you have created the subset directory (e.g. by running 'python sub.py') first.")
    sys.exit(1)

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

# Get all identity directories
identities = [d for d in os.listdir(SUBSET_PATH) 
              if os.path.isdir(os.path.join(SUBSET_PATH, d))]

print(f"Found {len(identities)} identities")
print(f"Reducing to {IMAGES_PER_PERSON} images per identity...\n")

for identity in identities:
    identity_dir = os.path.join(SUBSET_PATH, identity)
    
    # Get all image files
    all_files = [f for f in os.listdir(identity_dir) 
                 if is_image_file(f)]
    
    original_count = len(all_files)
    
    if len(all_files) > IMAGES_PER_PERSON:
        # Select files to keep
        files_to_keep = set(random.sample(all_files, IMAGES_PER_PERSON))
        
        # Delete files not in the keep list
        for filename in all_files:
            if filename not in files_to_keep:
                file_path = os.path.join(identity_dir, filename)
                os.remove(file_path)
        
        print(f"{identity}: {original_count} → {IMAGES_PER_PERSON} images (deleted {original_count - IMAGES_PER_PERSON})")
    else:
        print(f"{identity}: {original_count} images (no change)")

print("\n✓ Successfully reduced dataset in-place!")