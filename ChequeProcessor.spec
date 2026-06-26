# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Cheque Processor portal (Windows, onedir).
# Build on a WINDOWS machine:  pyinstaller ChequeProcessor.spec
#
# Before building, run:  python build_tools/fetch_models.py   (downloads hf_models/)

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas, binaries, hiddenimports = [], [], []

# These libraries ship data files / dynamic submodules PyInstaller misses.
for pkg in ["torch", "torchvision", "transformers", "surya", "easyocr",
            "rapidocr_onnxruntime", "ultralytics", "gradio", "gradio_client",
            "safehttpx", "groovy", "tokenizers", "sentencepiece", "cv2"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:
        print(f"[spec] skip {pkg}: {e}")

# Bundle the app's own resources: detector weights, date model, base models.
datas += [
    ("backend", "backend"),
    ("models", "models"),
    ("hf_models", "hf_models"),
    ("config.py", "."),
    ("runtime_paths.py", "."),
    ("app.py", "."),
]

hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("surya")
hiddenimports += ["config", "runtime_paths", "app",
                  "backend.ml_models", "backend.ocr_engine"]

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tensorflow", "jax"],   # not needed; trims size
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ChequeProcessor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX often corrupts torch DLLs — keep off
    console=True,              # keep a console so errors are visible to you
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="ChequeProcessor",
)
