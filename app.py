import os
import cv2
import gradio as gr
from backend.ml_models import process_cheque as pc
from backend.ocr_engine import extract_text as et
from backend.ml_models import dl

c = """
.icon-button-wrapper.top-panel {
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s ease;
}
.image-container:hover .icon-button-wrapper.top-panel {
    opacity: 1;
    visibility: visible;
}
"""


def gb():
    return [
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None),
        gr.update(visible=False, value=None),
    ]


def ra():
    return [
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=None),
        gr.update(value=[]),
        gr.update(value=[]),
        gr.update(value=None),
        gr.update(value=""),
        gr.update(visible=False, value=None),
    ]


with gr.Blocks(theme=gr.themes.Soft(), css=c) as d:
    gr.Markdown("# Cheque Processing & OCR Portal")

    pic = gr.State(value={})
    eo = gr.Textbox(label="System Status", value="", interactive=False)

    with gr.Column() as p1:
        gr.Markdown("### Step 1: Upload Cheque and Select Models")
        ii = gr.Image(type="filepath", label="Upload Cheque Image")

        gr.Markdown("#### Word Segmentation")
        so = gr.CheckboxGroup(choices=["YOLO", "UNet"], show_label=False)

        gr.Markdown("#### Word Detection")
        do = gr.CheckboxGroup(choices=["Yolo26m", "Yolo8m", "RCNN"], show_label=False)

        pb = gr.Button("Process Cheque", variant="primary")

    with gr.Column(visible=False) as p2:
        gr.Markdown("### Step 2: Model Output Parallel Preview")

        with gr.Row():
            i1 = gr.Image(visible=False, show_label=False, container=False, interactive=False)
            i2 = gr.Image(visible=False, show_label=False, container=False, interactive=False)
            i3 = gr.Image(visible=False, show_label=False, container=False, interactive=False)

        with gr.Row():
            i4 = gr.Image(visible=False, show_label=False, container=False, interactive=False)
            i5 = gr.Image(visible=False, show_label=False, container=False, interactive=False)
            i6 = gr.Image(visible=False, show_label=False, container=False, interactive=False)

        ps = gr.CheckboxGroup(label="Select Images to Action", choices=[], visible=False)

        with gr.Row():
            ocr_selector = gr.CheckboxGroup(
                choices=["EasyOCR", "Surya", "TrOCR", "RapidOCR"],
                value=["RapidOCR"],
                label="Step 2a: Select OCR Engines for Comparison",
            )

        with gr.Row():
            ob = gr.Button("Proceed for OCR Extraction", variant="primary")
            db = gr.Button("Download Selected", variant="secondary")
            bb = gr.Button("Back to Model Selection")

        do_f = gr.File(label="Ready for Download", visible=False)
        orv = gr.Textbox(label="Extracted Text Output", visible=False, lines=10)
        unb = gr.Button("Upload New Cheque", variant="stop")

    pb.click(fn=pc, inputs=[ii, so, do], outputs=[p1, p2, ps, i1, i2, i3, i4, i5, i6, pic, eo])
    ob.click(fn=et, inputs=[ps, pic, ocr_selector], outputs=[orv])
    db.click(fn=dl, inputs=[ii, ps, pic], outputs=[do_f])
    bb.click(fn=gb, inputs=[], outputs=[p1, p2, orv, eo, do_f])
    unb.click(fn=ra, inputs=[], outputs=[p1, p2, ii, so, do, orv, eo, do_f])


if __name__ == "__main__":
    # Direct launch still works (python app.py); run.py is the preferred entry.
    d.launch(debug=True)
