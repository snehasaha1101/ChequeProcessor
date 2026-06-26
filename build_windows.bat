@echo off
REM ============================================================
REM  Build the Cheque Processor as a self-contained Windows app
REM  Run this on a WINDOWS machine with Python 3.10 or 3.11.
REM ============================================================

echo [1/5] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [2/5] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

echo [3/5] Downloading base models for offline bundling...
python build_tools\fetch_models.py

echo [4/5] Make sure your detector weights are in place:
echo        backend\models\rcnn\best_fasterrcnn_strategyA.pth
echo        backend\models\unet\m2_bdft_idrbt_finetuned.pth
echo        backend\models\yolo\cheque_yolo_v1.pt
echo        backend\models\yolo8m\v8m_transfer.pt
echo        backend\models\yolo26m\y26m_transfer.pt
echo    and the date model at  models\trocr_field_models\Date_noslash\
pause

echo [5/5] Freezing with PyInstaller...
pyinstaller ChequeProcessor.spec --noconfirm

echo.
echo ============================================================
echo  Done. Your app is in:  dist\ChequeProcessor\
echo  Zip that whole folder and send it to your supervisor.
echo  They unzip and double-click ChequeProcessor.exe
echo ============================================================
pause
