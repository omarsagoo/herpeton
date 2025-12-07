import io
import json
from pathlib import Path
from typing import List, Tuple

import open_clip
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from lightning.fabric import Fabric
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
from torch import nn


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "biomodel"
CLASS_MAP_PATH = BASE_DIR / "models" / "class_to_idx.json"
CLASS_METADATA_PATH = BASE_DIR / "models" / "class_metadata.json"
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
FAVICON_PATH = BASE_DIR.parent / "images" / "herpeton_logo.png"
DEFAULT_TOP_K = 3


def load_class_maps(path: Path) -> Tuple[dict, dict]:
    """Load class <-> index mappings."""
    with path.open("r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


class_to_idx, idx_to_class = load_class_maps(CLASS_MAP_PATH)

NUM_CLASSES = len(class_to_idx)

fabric = Fabric()

with CLASS_METADATA_PATH.open("r") as f:
    CLASS_METADATA = json.load(f)


class BioCLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.clip_model = clip_model
        embed_dim = clip_model.visual.output_dim
        self.head = nn.Linear(embed_dim, num_classes)

        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if not any(p.requires_grad for p in self.clip_model.parameters()) else torch.enable_grad():
            image_features = self.clip_model.encode_image(images)
        return self.head(image_features)


def build_model(num_classes: int):
    """Load BioCLIP backbone, attach classifier head, and restore checkpoint."""
    with fabric.init_module():
        clip_model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )
        model = BioCLIPClassifier(clip_model=clip_model, num_classes=num_classes, freeze_backbone=True)

    checkpoint = fabric.load(MODEL_PATH)
    model.load_state_dict(checkpoint["model"])
    model.to(fabric.device)
    model.eval()
    return model, preprocess_val


model_bio, bio_preprocess_val = build_model(NUM_CLASSES)


def _to_tensor(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes into a batched tensor on the correct device."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image") from exc

    tensor = bio_preprocess_val(image).unsqueeze(0)  # add batch dimension
    return tensor.to(fabric.device)


def predict_image(image_bytes: bytes, top_k: int = 1) -> List[Tuple[str, float]]:
    """Return the top-k class names and scores for an image."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    inputs = _to_tensor(image_bytes)

    with torch.no_grad():
        logits = model_bio(inputs)
        probs = F.softmax(logits, dim=1)
        scores, indices = probs.topk(k=min(top_k, NUM_CLASSES), dim=1)

    top_scores = scores[0].tolist()
    top_indices = indices[0].tolist()

    return [(idx_to_class[i], float(s)) for i, s in zip(top_indices, top_scores)]


def format_predictions(predictions: List[Tuple[str, float]]) -> dict:
    """Shape predictions for the HTTP responses."""
    enriched = []
    for label, score in predictions:
        meta = CLASS_METADATA.get(label, {})
        enriched.append(
            {
                "display_name": meta.get("common_name") or meta.get("scientificName") or label,
                "common_name": meta.get("common_name"),
                "scientific_name": meta.get("scientificName") or label,
                "label": label,
                "family": meta.get("family"),
                "genus": meta.get("genus"),
                "order": meta.get("order"),
                "class": meta.get("class"),
                "kingdom": meta.get("kingdom"),
                "score": score,
            }
        )
    return {"predictions": enriched}


app = FastAPI(title="Herpeton Reptile Classifier")


@app.get("/", response_class=FileResponse)
def index():
    """Serve the landing page UI."""
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=500, detail="Landing page template is missing.")
    return FileResponse(TEMPLATE_PATH)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve the site favicon (tiny reptile logo)."""
    if not FAVICON_PATH.exists():
        raise HTTPException(status_code=404, detail="Favicon not found.")
    return FileResponse(FAVICON_PATH)


async def _predict_endpoint(file: UploadFile, top_k: int):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        predictions = predict_image(image_bytes, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return format_predictions(predictions)


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = DEFAULT_TOP_K):
    """Accept an uploaded image and return top predictions."""
    return await _predict_endpoint(file, top_k)


@app.post("/predict_frame")
async def predict_frame(file: UploadFile = File(...), top_k: int = DEFAULT_TOP_K):
    """Accept a live-frame image upload and return predictions."""
    return await _predict_endpoint(file, top_k)
