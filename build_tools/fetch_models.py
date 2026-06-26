"""
Run this ONCE on your build machine, with internet, before freezing.
It downloads the base OCR models into ./hf_models so they can be bundled
into the frozen app and used offline on the supervisor's machine.
"""
import os
os.makedirs("hf_models", exist_ok=True)
os.environ["HF_HOME"] = os.path.abspath("hf_models")

from huggingface_hub import snapshot_download

print("Downloading TrOCR-large-handwritten ...")
snapshot_download(repo_id="microsoft/trocr-large-handwritten",
                  local_dir="hf_models/trocr-large-handwritten")

print("Caching Surya foundation + recognition + detection models ...")
try:
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    f = FoundationPredictor()
    RecognitionPredictor(f)
    DetectionPredictor()
    print("Surya cached under hf_models/")
except Exception as e:
    print("Surya caching note:", e)

print("\nDone. hf_models/ now holds the base models for offline bundling.")
