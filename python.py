import os

# Check if folders exist
base_path = r"E:\Projects\Medical Image Classification for Pneumonia Detection\data\raw\chest_xray"

folders_to_check = [
    "train/NORMAL",
    "train/PNEUMONIA",
    "val/NORMAL",
    "val/PNEUMONIA",
    "test/NORMAL",
    "test/PNEUMONIA"
]

print("Checking dataset structure...\n")
for folder in folders_to_check:
    path = os.path.join(base_path, folder)
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    
    if exists:
        count = len([f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{status} {folder:<20} ({count} images)")
    else:
        print(f"{status} {folder:<20} (NOT FOUND)")