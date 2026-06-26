
help me show the two images in parallelA handwritten-cheque field extraction portal. Detects the four key fields on an Indian bank cheque — **Date, Payee Name, Amount in Words, Amount in Numbers** — then reads each with the OCR engine of your choice and shows the outputs side by side for comparison.


<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/5d8bf48d-0a24-4f73-a934-1c9f4c6d1d1a" />
<img width="959" height="538" alt="image" src="https://github.com/user-attachments/assets/f140ce07-db04-478a-970e-611948c8ae3b" />



## What it does

Upload a cheque image, pick a detector and one or more OCR engines, and the portal:
1. **Detects** the four field bounding boxes with your chosen detector
2. **Crops** each field with padding
3. **Reads** each field with the OCR engines you selected, in parallel
4. **Displays** every engine's output for every field, alongside the detection visualisation

This is the live counterpart to the evaluation pipeline used to benchmark detectors and OCR engines on the **IDRBT cheque dataset** and an internal SBI scanned set.

## Models included

**Detectors / segmenters**
- **Faster R-CNN** (ResNet-50 FPN) — best CER overall, recommended default
- **YOLOv8m** and **YOLOv8m fine-tuned** — faster but slightly weaker on dense Axis-Bank microprint cheques
- **YOLOv8 word-segmentation** and **U-Net** — for full-cheque segmentation comparison

**OCR engines**
- **TrOCR-large-handwritten** (base) — strong on names and amount-in-words
- **TrOCR with field routing** — the Date field is read by a synthetic no-slash specialist that fixed base TrOCR's date hallucination, while every other field uses base TrOCR
- **Surya OCR** — best on digit fields (Date, Amount-in-Numbers)
- **EasyOCR**, **RapidOCR** — additional baselines for comparison

The recommended deployable configuration that came out of the eval is **RCNN + Surya (digits) + base TrOCR (names/words)** with the no-slash Date specialist as a TrOCR alternative.



## Install (local)

```bash
git clone https://github.com/<your-username>/ChequeProcessor.git
cd ChequeProcessor
python -m venv .venv
.venv\Scripts\activate              # Windows
source .venv/bin/activate           # Mac/Linux
pip install -r requirements.txt
```

Tested on Python 3.10 and 3.11. GPU optional but recommended (TrOCR-large on CPU is slow).

## Download weights

The model weights are not in the git repo. Download them from the project's **GitHub Releases** page and place them as follows:

```
backend/models/rcnn/best_fasterrcnn_strategyA.pth
backend/models/unet/m2_bdft_idrbt_finetuned.pth
backend/models/yolo/cheque_yolo_v1.pt
backend/models/yolo8m/v8m_transfer.pt
backend/models/yolo26m/y26m_transfer.pt
models/trocr_field_models/Date_noslash/   (folder with config.json + safetensors)
```

The base TrOCR-large, Surya, EasyOCR, and RapidOCR models download themselves from Hugging Face on first run.

## Run

```bash
python run.py                   # http://127.0.0.1:7860
python run.py --share           # temporary public link
python run.py --port 8080
```

First launch takes 30–60 s while models load. Subsequent launches are fast.

## Repository layout

```
app.py                          Gradio UI
run.py                          launcher (host / port / share)
config.py                       path resolution for all models
backend/
  ml_models.py                  detector routing
  ocr_engine.py                 OCR engines + field-routing for TrOCR
  models/
    rcnn/      model.py + .pth
    unet/      model.py + .pth
    yolo/      model.py + .pt
    yolo8m/    model.py + .pt
    yolo26m/   model.py + .pt
```

## Build a standalone Windows app (optional)

See `BUILD_WINDOWS.md`. Run `build_windows.bat` on a Windows machine — it produces a `dist\ChequeProcessor\` folder containing `ChequeProcessor.exe` and all dependencies that you can zip and send to a non-technical user. They double-click the exe; no Python, no install.

## Hosted demo?

This project is not hosted publicly. The dependency stack (PyTorch + TrOCR-large + Surya) needs more RAM and disk than free hosting tiers provide. The intended deployment is local (clone-and-run) or a frozen Windows folder for distribution to colleagues.

## Datasets

- **IDRBT Cheque Image Dataset** — 112 real Indian bank cheques (bank-stratified 50/50 split, seed 42)
- **SBI internal scanned set** — held-out validation
- **Synthetic cheque generator** — used for OCR fine-tuning experiments (see eval report)

## License

MIT. See `LICENSE`.

## Acknowledgements

R&D internship project at **IIT (ISM) Dhanbad** under Prof. Soumen Bag. IDRBT for the cheque dataset.
