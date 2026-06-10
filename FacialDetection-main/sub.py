import os
import shutil
import random

# Select 50 random identities from the dataset
all_identities = os.listdir(r'C:\Users\HP\Downloads\archive\train')
selected_identities = random.sample(all_identities, 25)

# Copy to new directory
for identity in selected_identities:
    shutil.copytree(rf'C:\Users\HP\Downloads\archive\train\{identity}', 
                    f'D:\ACMClub\CNNProject/subset/{identity}')