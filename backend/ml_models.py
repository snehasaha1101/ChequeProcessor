
from PIL import Image
import numpy as np
import gradio as gr
import cv2
import os

from backend.models.yolo.model import run_yolo
from backend.models.unet.model import run_unet
from backend.models.rcnn.model import run_rcnn
from backend.models.yolo26m.model import run_yolo_ft   # was yolo_finetuned
from backend.models.yolo8m.model  import run_yolo_rt    # was yolo_retrained

# Updated routing map with the dedicated functions
mod_map = {
    "YOLO": run_yolo,
    "UNet": run_unet,
    "Yolo26m": run_yolo_ft,
    "Yolo8m": run_yolo_rt,
    "RCNN": run_rcnn
}

def dl(p, s, st):
    if not s or not st:
        return gr.update(visible=False, value=None)

    bn = os.path.splitext(os.path.basename(p))[0]
    e = os.path.splitext(p)[1]
    res = []

    for x in s:
        n = x.replace("Output: ", "")
        if n in st:
            arr = st[n].get('d')
            if arr is not None:
                f = f"{bn}_{n.replace(' ', '_')}{e}"
                cv2.imwrite(f, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                res.append(f)

    return gr.update(value=res, visible=True)

def process_cheque(p, m_seg, m_det):
    m = m_seg + m_det

    if not p or not m:
        yield [gr.update()] * 9 + [{}, "Error: Please upload an image and select at least one model."]
        return

    c = [(f"Output: {x}", x) for x in m]
    b = [gr.update(visible=False), gr.update(visible=True), gr.update(choices=c, value=[], visible=True)]

    st = {}
    imgs = [None] * 6

    def get_upd():
        u = []
        for i in range(6):
            if i < len(m):
                n = m[i]
                if imgs[i] is not None:
                    u.append(gr.update(value=imgs[i], visible=True, label=n, show_label=True, container=True))
                else:
                    u.append(gr.update(value=None, visible=False, label="", show_label=False, container=True))
            else:
                u.append(gr.update(value=None, visible=False, label="", show_label=False, container=True))
        return u

    yield b + get_upd() + [st, "Initializing..."]

    for i, mod in enumerate(m):
        if i >= 6:
            break

        yield b + get_upd() + [st, f"Running {mod}..."]

        with Image.open(p) as pil_image:
            og = np.array(pil_image.convert('RGB'))

        fn = mod_map.get(mod, run_yolo_ft)
        rgb, crp = fn(p, og)

        imgs[i] = rgb
        st[mod] = {'d': rgb, 'c': crp}

        yield b + get_upd() + [st, f"Finished {mod}."]

    yield b + get_upd() + [st, "Processing complete."]