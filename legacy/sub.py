# sub.py
# This script is part of a face recognition data pipeline.
# It selects a random subset of identities (folders) from the input dataset and copies them to a target directory.
# Input: A directory containing directories of different identities (e.g., 'archive/train').
# Output: A 'subset' directory containing only the randomly selected subset of identity directories.

import os
import shutil
import random
import sys

# Configuration variables
INPUT_DIR = r'archive/train'
OUTPUT_DIR = r'subset'
NUM_IDENTITIES = 25

# Basic error handling
if not os.path.exists(INPUT_DIR):
    print(f"Error: Expected input directory '{INPUT_DIR}' does not exist.")
    print("Please make sure you have downloaded the VGGFace2 dataset and placed it in the correct path.")
    sys.exit(1)

# Core logic
all_identities = os.listdir(INPUT_DIR)
# In case there are fewer identities than requested:
num_to_select = min(len(all_identities), NUM_IDENTITIES)
selected_identities = random.sample(all_identities, num_to_select)

# Copy to new directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
for identity in selected_identities:
    src_path = os.path.join(INPUT_DIR, identity)
    dst_path = os.path.join(OUTPUT_DIR, identity)
    if os.path.exists(dst_path):
        print(f"Destination path {dst_path} already exists, skipping...")
        continue
    shutil.copytree(src_path, dst_path)

print(f"Successfully copied {num_to_select} identities to '{OUTPUT_DIR}'.")