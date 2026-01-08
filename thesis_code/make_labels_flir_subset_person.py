import json
import os
from PIL import Image

# === INPUTS ===
IMG_DIR = r"C:\stage defensie\FLIR_ADAS_1_3\val\flir_subset_person_yolo\images"
ANN_FILE = r"C:\stage defensie\FLIR_ADAS_1_3\val\thermal_annotations.json"

# === OUTPUT ===
LABEL_DIR = r"C:\stage defensie\FLIR_ADAS_1_3\val\flir_subset_person_yolo\labels"
os.makedirs(LABEL_DIR, exist_ok=True)

# We evaluate person-only
TARGET_CATEGORY_NAME = "person"
YOLO_CLASS_ID = 0

#  collect subset images (stems) 
img_files = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
stem_to_filename = {os.path.splitext(f)[0]: f for f in img_files}
subset_stems = set(stem_to_filename.keys())

# load COCO annotations 
with open(ANN_FILE, "r") as f:
    coco = json.load(f)

#  category name -> id (robust) 
cat_name_to_id = {c["name"].lower(): c["id"] for c in coco.get("categories", [])}
if TARGET_CATEGORY_NAME not in cat_name_to_id:
    raise ValueError(
        f"Category '{TARGET_CATEGORY_NAME}' not found in JSON. "
        f"Found: {list(cat_name_to_id.keys())}"
    )
person_cat_id = cat_name_to_id[TARGET_CATEGORY_NAME]

#  map COCO image_id -> subset filename by matching stem of basename 
id_to_subset_filename = {}
for img in coco["images"]:
    coco_basename = os.path.basename(img["file_name"])  # drop subfolders
    coco_stem = os.path.splitext(coco_basename)[0]      # drop extension
    if coco_stem in subset_stems:
        id_to_subset_filename[img["id"]] = stem_to_filename[coco_stem]

#  create/clear label files for all subset images 
for stem in subset_stems:
    with open(os.path.join(LABEL_DIR, stem + ".txt"), "w") as f:
        pass

written = 0
skipped_missing_img = 0

#  write only person annotations 
for ann in coco["annotations"]:
    if ann.get("category_id") != person_cat_id:
        continue

    img_id = ann.get("image_id")
    if img_id not in id_to_subset_filename:
        continue

    fn = id_to_subset_filename[img_id]
    img_path = os.path.join(IMG_DIR, fn)
    if not os.path.exists(img_path):
        skipped_missing_img += 1
        continue

    # COCO bbox: [x_min, y_min, width, height] in pixels
    x, y, w, h = ann["bbox"]

    # Get image size
    with Image.open(img_path) as im:
        W, H = im.size

    # Convert to YOLO normalized xywh
    x_c = (x + w / 2) / W
    y_c = (y + h / 2) / H
    w_n = w / W
    h_n = h / H

    # Write: class x_center y_center width height
    out_txt = os.path.join(LABEL_DIR, os.path.splitext(fn)[0] + ".txt")
    with open(out_txt, "a") as f:
        f.write(f"{YOLO_CLASS_ID} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
    written += 1

print("Images in subset folder:", len(img_files))
print("Images matched in JSON:", len(id_to_subset_filename))
print("Person boxes written:", written)
print("Skipped missing images:", skipped_missing_img)
print("Labels dir:", LABEL_DIR)
