"""
Resolves resource locations for BOTH source runs and frozen (PyInstaller) runs,
and forces fully-offline model loading so the app never reaches the network.

Import this FIRST, before transformers/surya, so the env vars take effect.
"""
import os
import sys


def base_dir():
    # When frozen by PyInstaller (onedir), resources sit next to the exe in _MEIPASS.
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


BASE = base_dir()

# Point Hugging Face + Surya at the bundled model cache and go offline.
_HF = os.path.join(BASE, "hf_models")
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Base TrOCR: use the bundled snapshot folder if present, else the hub id.
_TROCR_LOCAL = os.path.join(_HF, "trocr-large-handwritten")
if os.path.isdir(_TROCR_LOCAL):
    os.environ.setdefault("TROCR_MODEL", _TROCR_LOCAL)

# No-slash Date specialist: bundled under models/ in the frozen app.
_DATE_PARENT = os.path.join(BASE, "models", "trocr_field_models")
if os.path.isdir(_DATE_PARENT):
    os.environ.setdefault("TROCR_FIELD_MODELS_DIR", _DATE_PARENT)
