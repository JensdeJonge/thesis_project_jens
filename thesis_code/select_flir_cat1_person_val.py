import json
import os
import shutil

# Root directory of the FLIR dataset
ROOT = r"C:\stage defensie\FLIR_ADAS_1_3"

# Dataset split to use: "val" or "train"
SPLIT = "val"

# Paths within the dataset
ANN_FILE = os.path.join(ROOT, SPLIT, "thermal_annotations.json")
OUT_DIR  = os.path.join(ROOT, SPLIT, "thermal_8_bit_cat1_person")

# Create output directory if it does not exist
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Reading annotations from: {ANN_FILE}")
with open(ANN_FILE, "r") as f:
    data = json.load(f)

# In the FLIR dataset: category_id 1 corresponds to 'People'
person_cat_id = 1
print(f"Using category_id (People): {person_cat_id}")

# Collect all image_ids that contain at least one annotation
# with category_id = 1 (person)
person_image_ids = {
    ann["image_id"]
    for ann in data["annotations"]
    if ann["category_id"] == person_cat_id
}

print(f"Total number of images containing category 1 (People): {len(person_image_ids)}")

# Map image_id to file_name
# Note: file_name already contains a subpath, e.g. 'thermal_8_bit/FLIR_10222.jpeg'
id2file = {img["id"]: img["file_name"] for img in data["images"]}

missing = 0
copied = 0

# Iterate over all images that contain at least one person annotation
for img_id in person_image_ids:
    file_name = id2file[img_id]

    # The file_name already includes 'thermal_8_bit/...'
    # so we only prepend ROOT and SPLIT to construct the full source path
    src = os.path.join(ROOT, SPLIT, file_name)
    dst = os.path.join(OUT_DIR, os.path.basename(file_name))

    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1
    else:
        print(f"Warning: source image not found: {src}")
        missing += 1

print("Done!")
print(f"Successfully copied images: {copied}")
print(f"Thermal images containing category 1 (People) are stored in: {OUT_DIR}")
if missing:
    print(f"Warning: {missing} images could not be found.")
