# Building the Windows app (for you, on your Windows machine)

This produces a folder your supervisor unzips and runs by double-clicking
`ChequeProcessor.exe` — no Python, no internet, no setup on their end.

## Prerequisites (your build machine)
- Windows 10/11
- Python 3.10 or 3.11 (NOT 3.12+ — some deps lag)  — "Add to PATH" at install
- ~20 GB free disk (build artifacts + bundled models are large)
- Internet (only while building, to fetch the base models)

## Steps
1. Put this whole folder on your Windows machine.
2. Place your trained weights:
   ```
   backend\models\rcnn\best_fasterrcnn_strategyA.pth
   backend\models\unet\m2_bdft_idrbt_finetuned.pth
   backend\models\yolo\cheque_yolo_v1.pt
   backend\models\yolo8m\v8m_transfer.pt
   backend\models\yolo26m\y26m_transfer.pt
   models\trocr_field_models\Date_noslash\   (the no-slash date model)
   ```
3. Double-click **build_windows.bat** (or run it in a terminal).
   It makes a venv, installs everything, downloads the base models into
   `hf_models\`, then freezes the app.
4. When it finishes, your app is in **dist\ChequeProcessor\**.
5. Right-click that folder -> Send to -> Compressed (zipped) folder.
   Send the zip to your supervisor with SUPERVISOR_README.txt.

## Expected result
- `dist\ChequeProcessor\` will be several GB (torch + models). Normal.
- First launch on any machine takes ~30–60s (loading models). Normal.

## If the build fails
The console stays open (console=True in the spec) so you can read the error.
The usual culprits and fixes:
- "module not found" at runtime -> add it to `hiddenimports` in ChequeProcessor.spec
- a missing data file (transformers/surya) -> already handled by collect_all,
  but if one slips through, add it to `datas` in the spec
- torch DLL error -> make sure `upx=False` (it is) and you didn't install a
  CUDA torch build you don't have drivers for; CPU torch is fine and smaller
- onnxruntime / easyocr issues -> confirm rapidocr_onnxruntime and easyocr
  imported cleanly when you ran the app from source first

## Sanity check BEFORE freezing
Run it from source once on the build machine to confirm it works:
```
.venv\Scripts\activate.bat
set HF_HOME=%CD%\hf_models
python run.py
```
If that opens and reads a cheque, the freeze will almost certainly work too.
