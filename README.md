
- `eval_flir_person_baseline.py`  
  Computes precision, recall, and F1-score for person detection on the FLIR validation set.

- `make_labels_flir_subset_person.py`  
  Converts FLIR COCO-style annotations to YOLO ground-truth labels.

- `select_flir_cat1_person_val.py`  
  Creates a person-present validation subset from the FLIR dataset.

## Dataset

The FLIR ADAS v1.3 dataset is not included in this repository due to licensing
and privacy restrictions. The dataset must be obtained directly from FLIR and placed
in the expected directory structure.

## Baseline Configuration

- Model: YOLOv7 (COCO-pretrained)
- Task: Person detection
- Dataset: FLIR ADAS v1.3 (thermal images)
- IoU threshold: 0.5
- Confidence threshold: 0.25

## Acknowledgements

This project is based on the YOLOv7 implementation by Wong et al.
The original repository can be found at:

https://github.com/WongKinYiu/yolov7

The original codebase was used as a foundation and extended with
custom preprocessing and evaluation scripts for thermal imagery.
