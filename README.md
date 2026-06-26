# Cheque Processor

A handwritten-cheque field extraction portal. It detects the four key handwritten fields on an Indian bank cheque — **Date, Payee Name, Amount in Words, and Amount in Numbers** — then reads each field using one or more OCR engines and displays the extracted text side by side for comparison.

<table>
<tr>
<td width="50%" align="center">

### Application Interface

<img src="https://github.com/user-attachments/assets/5d8bf48d-0a24-4f73-a934-1c9f4c6d1d1a" width="100%">

</td>

<td width="50%" align="center">

### OCR Output Comparison

<img src="https://github.com/user-attachments/assets/f140ce07-db04-478a-970e-611948c8ae3b" width="100%">

</td>
</tr>
</table>

---

# Features

Upload a cheque image, choose a detector and one or more OCR engines, and the application automatically:

1. **Detects** the four handwritten cheque fields.
2. **Crops** every detected field with configurable padding.
3. **Runs OCR** using one or multiple OCR engines simultaneously.
4. **Displays** the extracted text from every OCR engine side by side.
5. **Visualizes** the detected field bounding boxes on the original cheque.

The application serves as the interactive counterpart of the evaluation pipeline used to benchmark various detectors and OCR engines on the **IDRBT Cheque Dataset** and an internal SBI scanned cheque dataset.

---

# Models Included

## Detection / Segmentation Models

| Model | Purpose |
|--------|---------|
| Faster R-CNN (ResNet-50 FPN) | Default detector with the best overall OCR performance |
| YOLOv8m | Fast handwritten field detector |
| YOLOv8m (Fine-tuned) | Improved YOLO detector after transfer learning |
| YOLO26m | Additional segmentation comparison |
| U-Net | Handwritten region segmentation |

---

## OCR Engines

| OCR Engine | Best Use |
|------------|----------|
| TrOCR-large-Handwritten | Names and Amount-in-Words |
| TrOCR Field Routing | Date specialist + Base TrOCR |
| Surya OCR | Dates and numeric amounts |
| EasyOCR | Baseline comparison |
| RapidOCR | Lightweight OCR baseline |

### Recommended Configuration

**Faster R-CNN + Surya OCR (digits) + Base TrOCR (names & words)**

This combination achieved the best overall recognition accuracy during evaluation.

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/ChequeProcessor.git
cd ChequeProcessor
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

**Tested on**

- Python 3.10
- Python 3.11

GPU acceleration is optional but recommended when using TrOCR-large.

---

# Download Model Weights

The repository contains only the source code.

Download the trained models from the project's **GitHub Releases** page and place them in the following locations:

```
backend/
└── models/
    ├── rcnn/
    │   └── best_fasterrcnn_strategyA.pth
    │
    ├── unet/
    │   └── m2_bdft_idrbt_finetuned.pth
    │
    ├── yolo/
    │   └── cheque_yolo_v1.pt
    │
    ├── yolo8m/
    │   └── v8m_transfer.pt
    │
    └── yolo26m/
        └── y26m_transfer.pt

models/
└── trocr_field_models/
    └── Date_noslash/
```

The following OCR models download automatically from Hugging Face during first launch:

- TrOCR-large-Handwritten
- Surya OCR
- EasyOCR
- RapidOCR

---

# Running the Application

Default

```bash
python run.py
```

Public share

```bash
python run.py --share
```

Custom port

```bash
python run.py --port 8080
```

The first launch typically requires **30–60 seconds** while the AI models are loaded into memory.

---

# Repository Structure

```
ChequeProcessor/

├── app.py
├── run.py
├── config.py
├── runtime_paths.py
├── requirements.txt
├── build_windows.bat
├── BUILD_WINDOWS.md
│
├── backend/
│   ├── ml_models.py
│   ├── ocr_engine.py
│   └── models/
│       ├── rcnn/
│       ├── yolo/
│       ├── yolo8m/
│       ├── yolo26m/
│       └── unet/
│
└── build_tools/
```

---

# Build Standalone Windows Application

Run

```bash
build_windows.bat
```

This creates

```
dist/
└── ChequeProcessor/
    ├── ChequeProcessor.exe
    └── ...
```

The generated folder can be zipped and shared with users who do not have Python installed.

---

# Datasets

The project was developed and evaluated using:

- **IDRBT Cheque Image Dataset**
  - 112 real Indian bank cheques
  - Bank-stratified 50/50 train-test split

- **Internal SBI Scanned Dataset**
  - Used for validation and benchmarking

- **Synthetic Cheque Dataset**
  - Generated for OCR fine-tuning experiments

---

# Technologies Used

- Python
- OpenCV
- PyTorch
- Faster R-CNN
- YOLOv8
- U-Net
- Hugging Face Transformers
- TrOCR
- Surya OCR
- EasyOCR
- RapidOCR
- Gradio

---

# License

This project is released under the **MIT License**.

---

# Acknowledgements

This work was carried out during a research internship at **IIT (ISM) Dhanbad** under the guidance of **Prof. Soumen Bag**.

The authors also acknowledge **IDRBT** for providing the cheque image dataset used for experimentation and evaluation.
