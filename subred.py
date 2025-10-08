import os
import random

# Configuration
subset_path = r'D:\ACMClub\CNNProject\subset'
images_per_person = 150

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in image_extensions)

# Get all identity directories
identities = [d for d in os.listdir(subset_path) 
              if os.path.isdir(os.path.join(subset_path, d))]

print(f"Found {len(identities)} identities")
print(f"Reducing to {images_per_person} images per identity...\n")

for identity in identities:
    identity_dir = os.path.join(subset_path, identity)
    
    # Get all image files
    all_files = [f for f in os.listdir(identity_dir) 
                 if is_image_file(f)]
    
    original_count = len(all_files)
    
    if len(all_files) > images_per_person:
        # Select files to keep
        files_to_keep = set(random.sample(all_files, images_per_person))
        
        # Delete files not in the keep list
        for filename in all_files:
            if filename not in files_to_keep:
                file_path = os.path.join(identity_dir, filename)
                os.remove(file_path)
        
        print(f"{identity}: {original_count} → {images_per_person} images (deleted {original_count - images_per_person})")
    else:
        print(f"{identity}: {original_count} images (no change)")

print("\n✓ Successfully reduced dataset in-place!")