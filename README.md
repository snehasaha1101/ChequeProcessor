# Cheque Processing & OCR Portal

A local Gradio app that detects the four handwritten fields on a cheque
(Date, Name, Amount in Words, Amount in Numbers) and reads them with a choice
of OCR engines, side by side for comparison.

- **Detectors / segmenters:** RCNN (Faster R-CNN), Yolo8m, Yolo26m, YOLO-seg, U-Net
- **OCR engines:** EasyOCR, Surya, TrOCR, RapidOCR
- **Field routing:** TrOCR reads the Date field with a no-slash specialist
  (digit-only output) and every other field with base TrOCR-large; the
  Amount-in-Words output is snapped to a reference vocabulary with the
  cheque's original casing preserved.

## 1. Install

```bash
git clone <your-repo-url> cheque_processor
cd cheque_processor
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU is optional but strongly recommended (the app auto-detects CUDA and falls
back to CPU). TrOCR-large and Surya are heavy on CPU.

## 2. Model weights

**Detector weights ship in the repo**, next to the code that loads them:

```
backend/models/rcnn/best_fasterrcnn_strategyA.pth
backend/models/unet/m2_bdft_idrbt_finetuned.pth
backend/models/yolo/cheque_yolo_v1.pt
backend/models/yolo8m/v8m_transfer.pt
backend/models/yolo26m/y26m_transfer.pt
```

If you keep these out of git (they're large — see .gitignore), drop them back
into those folders after cloning.

**The one external model** is the no-slash Date TrOCR specialist. Put it at:

```
models/trocr_field_models/Date_noslash/
```

or point an env var at its parent folder:

```bash
export TROCR_FIELD_MODELS_DIR=/path/to/trocr_field_models
```

**Base TrOCR-large and Surya download themselves** from Hugging Face on first
run and cache in `~/.cache/huggingface` — you don't manage those files.

## 3. Run

```bash
python run.py                 # http://127.0.0.1:7860
python run.py --share         # temporary public link
python run.py --port 8080
```

## 4. Configuration

All paths resolve from `config.py`. Override without editing code:

| Env var                  | Default                                   | Meaning                                  |
|--------------------------|-------------------------------------------|------------------------------------------|
| `TROCR_MODEL`            | `microsoft/trocr-large-handwritten`       | base TrOCR checkpoint                    |
| `TROCR_FIELD_MODELS_DIR` | `./models/trocr_field_models`             | folder holding `Date_noslash/`           |

## Notes

- The detectors expect their weight files by the exact names above. If you
  renamed any, update the corresponding `backend/models/<name>/model.py`.
- First run is slow: model downloads + lazy load. Subsequent runs are fast.
- On a 15 GB GPU, selecting many engines at once can exhaust memory; compare
  two or three at a time if you hit CUDA OOM.
