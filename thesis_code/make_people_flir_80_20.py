import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# PEOPLE-FLIR creator (FLIR ADAS v1.3)
# - Merge train + val
# - Filter images that contain at least one PERSON
# - Keep ONLY PERSON annotations
# - Convert COCO JSON bboxes -> YOLO .txt labels (class 0 = person)
# - Create a reproducible 80/20 split (train/test)
# - Write output to: C:\stage defensie\PEOPLE_FLIR
# ============================================================

# --------------------
# PATHS (YOUR PATHS)
# --------------------
TRAIN_IMG_DIR = Path(r"C:\stage defensie\FLIR_ADAS_1_3\train\thermal_8_bit")
VAL_IMG_DIR   = Path(r"C:\stage defensie\FLIR_ADAS_1_3\val\thermal_8_bit")

TRAIN_JSON = Path(r"C:\stage defensie\FLIR_ADAS_1_3\train\thermal_annotations.json")
VAL_JSON   = Path(r"C:\stage defensie\FLIR_ADAS_1_3\val\thermal_annotations.json")

OUT_ROOT = Path(r"C:\stage defensie\PEOPLE_FLIR")  # your empty output folder

# --------------------
# SETTINGS
# --------------------
TRAIN_FRACTION = 0.80
SEED = 42
MIN_PERSONS_PER_IMAGE = 1  # person-present filter

# Only accept these image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# HELPERS
# ============================================================

def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_out_dirs():
    (OUT_ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images" / "test").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / "test").mkdir(parents=True, exist_ok=True)

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def write_data_yaml():
    yaml = f"""train: {str((OUT_ROOT / "images" / "train").resolve())}
val: {str((OUT_ROOT / "images" / "test").resolve())}

nc: 1
names: ['person']
"""
    write_text(OUT_ROOT / "data.yaml", yaml)

def is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXTS

def find_person_category_id(coco: dict) -> int:
    """
    Find the COCO category id where name == "person" (case-insensitive).
    """
    for cat in coco.get("categories", []):
        if str(cat.get("name", "")).lower() == "person":
            return int(cat["id"])
    # If not found, show available names for debugging
    names = [c.get("name", None) for c in coco.get("categories", [])]
    raise ValueError(f"Could not find category name 'person'. Available categories: {names}")

def coco_bbox_to_yolo(bbox_xywh, img_w, img_h):
    """
    COCO bbox: [x_min, y_min, width, height] in pixels
    YOLO: x_center/img_w, y_center/img_h, w/img_w, h/img_h
    """
    x, y, w, h = bbox_xywh
    xc = x + w / 2.0
    yc = y + h / 2.0
    return (xc / img_w, yc / img_h, w / img_w, h / img_h)

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def resolve_image_path(file_name: str) -> Path | None:
    """
    COCO file_name might be basename or relative path. Try:
    - TRAIN_IMG_DIR / file_name
    - VAL_IMG_DIR / file_name
    - by basename in both dirs
    """
    fn = Path(file_name)

    # try as given (relative)
    p1 = TRAIN_IMG_DIR / fn
    if p1.exists():
        return p1
    p2 = VAL_IMG_DIR / fn
    if p2.exists():
        return p2

    # try basename only
    base = fn.name
    p3 = TRAIN_IMG_DIR / base
    if p3.exists():
        return p3
    p4 = VAL_IMG_DIR / base
    if p4.exists():
        return p4

    return None


# ============================================================
# MAIN
# ============================================================

def main():
    # Validate input paths
    for p in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_JSON, VAL_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    ensure_out_dirs()

    coco_train = load_coco(TRAIN_JSON)
    coco_val   = load_coco(VAL_JSON)

    # Person category id (should be consistent)
    person_id_train = find_person_category_id(coco_train)
    person_id_val   = find_person_category_id(coco_val)
    if person_id_train != person_id_val:
        print(f"WARNING: person category id differs: train={person_id_train}, val={person_id_val}")
    PERSON_CAT_ID = person_id_train

    # Merge images with safe keys to avoid id collisions across train/val
    merged_images = {}
    merged_annotations = []

    def ingest(coco: dict, source: str):
        for img in coco.get("images", []):
            key = (source, int(img["id"]))
            merged_images[key] = {
                "source": source,
                "id": int(img["id"]),
                "file_name": img["file_name"],
                "width": int(img["width"]),
                "height": int(img["height"]),
            }
        for ann in coco.get("annotations", []):
            merged_annotations.append({
                "source": source,
                "image_id": int(ann["image_id"]),
                "category_id": int(ann["category_id"]),
                "bbox": ann["bbox"],
                "iscrowd": int(ann.get("iscrowd", 0)),
            })

    ingest(coco_train, "train")
    ingest(coco_val, "val")

    # Group annotations per image
    anns_by_image = defaultdict(list)
    for ann in merged_annotations:
        key = (ann["source"], ann["image_id"])
        anns_by_image[key].append(ann)

    # Filter person-present and keep only person annotations
    person_present = []
    missing_image_files = 0
    skipped_no_person = 0
    skipped_nonimage = 0

    for key, imginfo in merged_images.items():
        file_name = imginfo["file_name"]
        if not is_image_file(file_name):
            skipped_nonimage += 1
            continue

        anns = anns_by_image.get(key, [])
        person_anns = [
            a for a in anns
            if a["category_id"] == PERSON_CAT_ID
        ]

        if len(person_anns) < MIN_PERSONS_PER_IMAGE:
            skipped_no_person += 1
            continue

        img_path = resolve_image_path(file_name)
        if img_path is None:
            missing_image_files += 1
            continue

        person_present.append((imginfo, img_path, person_anns))

    if not person_present:
        raise RuntimeError(
            "No person-present images found after filtering. "
            "Check that your JSON contains person annotations and paths match."
        )

    # 80/20 split
    random.seed(SEED)
    random.shuffle(person_present)

    n_total = len(person_present)
    n_train = int(round(TRAIN_FRACTION * n_total))

    train_set = person_present[:n_train]
    test_set  = person_present[n_train:]

    # Export split
    def export_split(split, split_name: str):
        for imginfo, img_path, person_anns in split:
            # Copy image
            dst_img = OUT_ROOT / "images" / split_name / img_path.name
            shutil.copy2(img_path, dst_img)

            # Build YOLO labels (class 0 = person)
            w = imginfo["width"]
            h = imginfo["height"]

            lines = []
            for ann in person_anns:
                xc, yc, bw, bh = coco_bbox_to_yolo(ann["bbox"], w, h)

                # clamp to [0,1] for safety
                xc = clamp01(xc); yc = clamp01(yc); bw = clamp01(bw); bh = clamp01(bh)

                if bw <= 0 or bh <= 0:
                    continue

                lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            dst_lbl = OUT_ROOT / "labels" / split_name / f"{img_path.stem}.txt"
            write_text(dst_lbl, "\n".join(lines) + ("\n" if lines else ""))

    export_split(train_set, "train")
    export_split(test_set, "test")
    write_data_yaml()

    # Summary
    print("==== PEOPLE-FLIR SUMMARY ====")
    print(f"Input train images dir:        {TRAIN_IMG_DIR}")
    print(f"Input val images dir:          {VAL_IMG_DIR}")
    print(f"Train JSON:                    {TRAIN_JSON}")
    print(f"Val JSON:                      {VAL_JSON}")
    print(f"Output root:                   {OUT_ROOT}")
    print(f"Person category id:            {PERSON_CAT_ID}")
    print(f"Kept person-present images:    {n_total}")
    print(f"Train split (80%):             {len(train_set)}")
    print(f"Test split (20%):              {len(test_set)}")
    print(f"Skipped (no person in image):  {skipped_no_person}")
    print(f"Missing image files:           {missing_image_files}")
    print(f"Skipped (non-image entries):   {skipped_nonimage}")
    print(f"YOLO data.yaml:                {OUT_ROOT / 'data.yaml'}")
    print("Done.")

if __name__ == "__main__":
    main()
