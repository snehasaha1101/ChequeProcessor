"""
Launcher for the Cheque Processor portal.

Usage:
    python run.py                 # local, opens http://127.0.0.1:7860
    python run.py --share         # also create a temporary public link
    python run.py --port 8080     # choose a port

Set model locations via environment variables (see config.py):
    TROCR_MODEL              base TrOCR (default microsoft/trocr-large-handwritten)
    TROCR_FIELD_MODELS_DIR   folder holding the Date_noslash specialist
"""
import argparse
import sys
import os

# Make "backend" importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    from app import d  # the gr.Blocks object
    d.launch(server_name=args.host, server_port=args.port,
             share=args.share, debug=args.debug)


if __name__ == "__main__":
    main()
