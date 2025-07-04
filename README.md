# Modular Object Detection on COCO (Unlabeled2017)

Detect objects in images using a pretrained Faster R‑CNN model from PyTorch, with a modular codebase structured for clarity, extensibility, and reusability.

---

## Overview

This project:

* Uses COCO’s `unlabeled2017` dataset (images only, no annotations).  
* Loads a Faster R‑CNN ResNet‑50 FPN model pretrained on COCO’s 80 categories.  
* Performs object detection, draws bounding boxes, and saves annotated results.  
* Is fully modularized for easy extension (training, pseudo‑labeling, fine‑tuning, etc.).

---

## Project Structure

```
coco_detector_modular/
├── config.py                # Global config: paths, thresholds, device, etc.
├── data/
│   ├── dataset.py           # Loads unlabeled COCO images
│   └── preprocessing.py     # Image preprocessing (transforms)
├── models/
│   └── detector.py          # Builds pretrained Faster R-CNN model
├── training/
│   ├── engine.py            # Training loops and utilities
│   └── train.py             # Fine‑tuning script (stub)
├── inference/
│   └── infer.py             # Runs inference and saves predictions
├── evaluation/
│   └── eval.py              # Placeholder for mAP/IoU evaluation
├── utils/
│   └── visualization.py     # Draws and saves detection results
├── requirements.txt         # Dependencies
└── README.md
```

---

## Setup

1. **Clone or unzip the project**

```bash
unzip coco_detector_modular.zip
cd coco_detector_modular
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download COCO unlabeled images (~20 GB)**

```bash
wget http://images.cocodataset.org/zips/unlabeled2017.zip
unzip unlabeled2017.zip -d coco/
```

Directory layout afterwards:

```
coco/
└── unlabeled2017/
    ├── 000000000001.jpg
    ├── 000000000002.jpg
    └── ...
```

---

## Inference (Object Detection)

```bash
python inference/infer.py
```

* Annotated images are saved to `./detections/`.  
* Console output lists detected classes and scores, e.g. `car 0.89`.

---

## Training / Fine‑tuning

`training/train.py` is a stub ready for your labeled dataset.

* Replace `CocoUnlabeledDataset` with a custom dataset that returns images **and** targets.  
* Provide `targets = [{"boxes": ..., "labels": ...}]` per image.  
* Extend loop logic in `training/engine.py`.

Run:

```bash
python training/train.py
```

Checkpoints are saved to `./checkpoints/`.

---

## Evaluation

`evaluation/eval.py` is a placeholder. Implement:

* COCO‑style mAP via `pycocotools`, or  
* Your own IoU / precision‑recall metrics.

---

## Customization Ideas

| Use case                       | How                                                               |
|--------------------------------|-------------------------------------------------------------------|
| Use a different detector       | Edit `models/detector.py` (e.g., `retinanet_resnet50_fpn`).       |
| Detect objects in video        | Read frames and reuse `inference/` logic.                         |
| Adjust confidence threshold    | Change `CONF_THRESH` in `config.py`.                              |
| Deploy with Streamlit / Gradio | Wrap `inference` functions in a UI.                               |

---

## Model Info

* **Architecture**: Faster R‑CNN with ResNet‑50 backbone + FPN  
* **Pretraining data**: 80‑class COCO dataset  
* **Output**: Bounding boxes with class labels and confidence scores

---

## Example Output

```
[12/123456] 000000394812.jpg
    person 0.98 [35.0, 22.4, 300.1, 450.2]
    dog    0.91 [120.2, 160.3, 300.5, 480.1]
```

---

## License

MIT. COCO dataset is licensed under its own terms.

---

## Questions or Help

Need guidance on training, data integration, or deployment? Feel free to reach out.
