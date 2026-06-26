"""Frozen-app entrypoint. Importing runtime_paths first sets offline env vars."""
import runtime_paths  # noqa: F401  (must be first — configures offline model paths)
import os
import sys

sys.path.insert(0, runtime_paths.BASE)

import webbrowser
import threading
from app import d


def _open():
    webbrowser.open("http://127.0.0.1:7860")


if __name__ == "__main__":
    threading.Timer(2.5, _open).start()  # pop the browser once the server is up
    d.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
