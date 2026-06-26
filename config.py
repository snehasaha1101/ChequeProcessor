"""
Central configuration for the Cheque Processor app.

All model locations resolve from here so nothing in the codebase hardcodes an
absolute path. Detector weights ship inside backend/models/<name>/ and are
found relative to this file. The only external model is the no-slash Date
TrOCR specialist; point TROCR_FIELD_MODELS_DIR at the folder that contains its
"Date_noslash" subfolder (defaults to ./models/trocr_field_models).
"""
import os

# Repo root = directory that contains this file.
ROOT = os.path.dirname(os.path.abspath(__file__))

# Detector weights live next to their model.py inside backend/models/<name>/.
BACKEND_MODELS = os.path.join(ROOT, "backend", "models")

# Base OCR engines (downloaded + cached by transformers / surya on first use).
TROCR_MODEL = os.environ.get("TROCR_MODEL", "microsoft/trocr-large-handwritten")

# The one fine-tuned model we keep: the synthetic no-slash Date specialist.
# Override with the env var if you store it elsewhere.
TROCR_FIELD_MODELS_DIR = os.environ.get(
    "TROCR_FIELD_MODELS_DIR",
    os.path.join(ROOT, "models", "trocr_field_models"),
)
DATE_MODEL_DIR = os.path.join(TROCR_FIELD_MODELS_DIR, "Date_noslash")


def weight(model_name, filename):
    """Absolute path to a detector weight bundled under backend/models/<name>/."""
    return os.path.join(BACKEND_MODELS, model_name, filename)
