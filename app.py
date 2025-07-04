# streamlit_app.py
"""Streamlit demo for object detection with Faster R‑CNN (COCO).

* Upload an image (JPG / PNG / BMP).
* The pretrained detector finds objects and draws bounding boxes.
* Category names come directly from the COCO weights—no `classes.txt` needed.
"""

import io
from pathlib import Path
from typing import List

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes

# Local import: make sure your project has models/detector.py
from models.detector import build_detector  # returns (model, categories)

# ----------------------------------------------------------------------------
# Config & helpers
# ----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TO_TENSOR = transforms.ToTensor()  # converts [0‒255] PIL → [0‒1] float tensor
TO_PIL = transforms.ToPILImage()

def run_detector(img: Image.Image, model: torch.nn.Module, conf: float):
    """Run model → filter predictions above *conf* (0–1)."""
    model.eval()
    tensor = TO_TENSOR(img).to(DEVICE)
    with torch.inference_mode():
        preds = model([tensor])[0]
    keep = preds["scores"] >= conf
    return {k: v[keep].cpu() for k, v in preds.items()}


def draw_results(img: Image.Image, preds: dict, categories: List[str]):
    """Return a PIL image with boxes & labels drawn."""
    if len(preds["boxes"]) == 0:
        return img
    labels = [f"{categories[l]} {s:.2f}" for l, s in zip(preds["labels"], preds["scores"])]
    tensor = TO_TENSOR(img)
    boxed = draw_bounding_boxes(
        (tensor * 255).byte(), boxes=preds["boxes"], labels=labels, width=3
    )
    return TO_PIL(boxed)

# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="COCO Object Detector", layout="centered")
st.title("COCO Object Detection Demo")
st.markdown(
    "Upload an image and the **Faster R‑CNN ResNet‑50 FPN** model pretrained on COCO will highlight detected objects."
)

@st.cache_resource(show_spinner=False)
def load_model():
    model, categories = build_detector()
    return model, categories

with st.spinner("Loading model …"):
    model, categories = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
conf_thres = st.slider("Confidence threshold", 0.05, 1.0, 0.3, 0.05)

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Original image", use_column_width=True)

    if st.button("Detect objects", type="primary"):
        with st.spinner("Running detector …"):
            preds = run_detector(image, model, conf_thres)
            result_img = draw_results(image, preds, categories)

        st.subheader("Detections")
        st.image(result_img, use_column_width=True)

        if len(preds["boxes"]):
            st.write("**Objects found:**")
            for label, score in zip(preds["labels"], preds["scores"]):
                st.write(f"- {categories[label]} ({score:.2f})")
        else:
            st.info("No objects above the confidence threshold.")
else:
    st.info("Please upload an image to begin.")
