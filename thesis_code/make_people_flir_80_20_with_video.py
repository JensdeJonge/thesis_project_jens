import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# PEOPLE-FLIR creator (FLIR ADAS v1.3) INCLUDING VIDEO SUBSET
# - Merge train + val + video
# - Filter images that contain at least one PERSON (category name 'person')
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
VIDEO_IMG_DIR = Path(r"C:\stage defensie\FLIR_ADAS_1_3\video\thermal_8_bit")

TRAIN_JSON = Path(r"C:\stage defensie\FLIR_ADAS_1_3\train\thermal_annotations.json")
VAL_JSON   = Path(r"C:\stage defensie\FLIR_ADAS_1_3\val\thermal_annotations.json")
VIDEO_JSON = Path(r"C:\stage defensie\FLIR_ADAS_1_3\video\thermal_annotations.json")

OUT_ROOT = Path(r"C:\stage defensie\PEOPLE_FLIR")  # output folder

# --------------------
# SETTINGS
# --------------------
TRAIN_FRACTION = 0.80
SEED = 42
MIN_PERSONS_PER_IMAGE = 1

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
    for cat in coco.get("categories", []):
        if str(cat.get("name", "")).lower() == "person":
            return int(cat["id"])
    names = [c.get("name", None) for c in coco.get("categories", [])]
    raise ValueError(f"Could not find category name 'person'. Available categories: {names}")

def coco_bbox_to_yolo(bbox_xywh, img_w, img_h):
    x, y, w, h = bbox_xywh
    xc = x + w / 2.0
    yc = y + h / 2.0
    return (xc / img_w, yc / img_h, w / img_w, h / img_h)

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def resolve_image_path(source: str, file_name: str) -> Path | None:
    """
    Resolve image path based on source split (train/val/video).
    """
    fn = Path(file_name)

    if source == "train":
        base_dir = TRAIN_IMG_DIR
    elif source == "val":
        base_dir = VAL_IMG_DIR
    elif source == "video":
        base_dir = VIDEO_IMG_DIR
    else:
        return None

    p = base_dir / fn
    if p.exists():
        return p

    # try basename only
    p2 = base_dir / fn.name
    if p2.exists():
        return p2

    return None


# ============================================================
# MAIN
# ============================================================

def main():
    for p in [TRAIN_IMG_DIR, VAL_IMG_DIR, VIDEO_IMG_DIR, TRAIN_JSON, VAL_JSON, VIDEO_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    ensure_out_dirs()

    coco_train = load_coco(TRAIN_JSON)
    coco_val   = load_coco(VAL_JSON)
    coco_video = load_coco(VIDEO_JSON)

    # person id should be consistent; warn if not
    pid_train = find_person_category_id(coco_train)
    pid_val   = find_person_category_id(coco_val)
    pid_video = find_person_category_id(coco_video)

    if len({pid_train, pid_val, pid_video}) != 1:
        print(f"WARNING: person category id differs: train={pid_train}, val={pid_val}, video={pid_video}")

    PERSON_CAT_ID = pid_train

    # Merge with safe keys to avoid id collisions
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
            })

    ingest(coco_train, "train")
    ingest(coco_val, "val")
    ingest(coco_video, "video")

    # Group annotations per image
    anns_by_image = defaultdict(list)
    for ann in merged_annotations:
        key = (ann["source"], ann["image_id"])
        anns_by_image[key].append(ann)

    # Filter person-present
    person_present = []
    missing_image_files = 0
    skipped_no_person = 0
    skipped_nonimage = 0

    for key, imginfo in merged_images.items():
        source = imginfo["source"]
        file_name = imginfo["file_name"]

        if not is_image_file(file_name):
            skipped_nonimage += 1
            continue

        anns = anns_by_image.get(key, [])
        person_anns = [a for a in anns if a["category_id"] == PERSON_CAT_ID]

        if len(person_anns) < MIN_PERSONS_PER_IMAGE:
            skipped_no_person += 1
            continue

        img_path = resolve_image_path(source, file_name)
        if img_path is None:
            missing_image_files += 1
            continue

        person_present.append((imginfo, img_path, person_anns))

    if not person_present:
        raise RuntimeError("No person-present images found. Check annotations and paths.")

    # 80/20 split
    random.seed(SEED)
    random.shuffle(person_present)

    n_total = len(person_present)
    n_train = int(round(TRAIN_FRACTION * n_total))

    train_set = person_present[:n_train]
    test_set  = person_present[n_train:]

    def export_split(split, split_name: str):
        for imginfo, img_path, person_anns in split:
            dst_img = OUT_ROOT / "images" / split_name / img_path.name
            shutil.copy2(img_path, dst_img)

            w = imginfo["width"]
            h = imginfo["height"]

            lines = []
            for ann in person_anns:
                xc, yc, bw, bh = coco_bbox_to_yolo(ann["bbox"], w, h)
                xc = clamp01(xc); yc = clamp01(yc); bw = clamp01(bw); bh = clamp01(bh)
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            dst_lbl = OUT_ROOT / "labels" / split_name / f"{img_path.stem}.txt"
            write_text(dst_lbl, "\n".join(lines) + ("\n" if lines else ""))

    export_split(train_set, "train")
    export_split(test_set, "test")
    write_data_yaml()

    print("==== PEOPLE-FLIR (WITH VIDEO) SUMMARY ====")
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
