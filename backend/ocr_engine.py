import gradio as gr
from PIL import Image
import numpy as np
import cv2
import os
import torch

from config import TROCR_MODEL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Engine handles are built on first use and cached here.
rapid = None
trocr_p = None
trocr_m = None
s_det = None
s_rec = None
foundation = None
easy = None
_current_field = None

try:
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    surya_available = True
except ImportError:
    surya_available = False


def _trocr_preprocess(img):
    """
    Minimal preprocessing for TrOCR.

    TrOCR was trained on natural handwriting images.
    Aggressive thresholding often destroys cursive strokes.
    """

    if isinstance(img, Image.Image):
        pi = img.convert("RGB")
    else:
        pi = Image.fromarray(
            np.asarray(img).astype(np.uint8)
        ).convert("RGB")

    w, h = pi.size

    # Upscale tiny crops.
    target_h = 128

    if h < target_h:
        scale = target_h / float(h)

        pi = pi.resize(
            (
                int(w * scale),
                target_h
            ),
            Image.Resampling.LANCZOS
        )

    # Small white border helps decoder.
    canvas = Image.new(
        "RGB",
        (pi.width + 20, pi.height + 20),
        (255, 255, 255)
    )

    canvas.paste(pi, (10, 10))

    return canvas


# --- OCR engines ------------------------------------------------------------
# Every engine receives an RGB image (numpy array or PIL Image) and returns text.

def run_easyocr(i):
    global easy
    try:
        if i is None or (hasattr(i, 'size') and i.size == 0):
            return ""
        import easyocr
        if easy is None:
            easy = easyocr.Reader(['en'], gpu=False)
        r = easy.readtext(np.asarray(i))
        if r:
            t = " ".join(x[1].strip() for x in r)
            return t if t not in ["-", "_", "___"] else ""
        return ""
    except Exception as e:
        return f"[EasyOCR error: {e}]"


def run_surya(i):
    global s_det, s_rec, foundation
    try:
        if i is None or (hasattr(i, 'size') and i.size == 0):
            return ""
        if not surya_available:
            return "[Surya not installed]"
        if foundation is None:
            foundation = FoundationPredictor()
        if s_rec is None:
            s_rec = RecognitionPredictor(foundation)
        # Input is already RGB; hand it straight to PIL, no channel swap.
        pi = Image.fromarray(np.asarray(i).astype(np.uint8))
        w, h = pi.size
        # The crop is already a single isolated field. Recognise it as one line
        # rather than letting Surya re-detect text inside it - on cheques that
        # re-detection transcribes the printed security microprint (the repeated
        # bank name) in the background and floods the output.
        try:
            p = s_rec([pi], bboxes=[[[0, 0, w, h]]])
        except Exception:
            if s_det is None:
                s_det = DetectionPredictor()
            p = s_rec([pi], det_predictor=s_det)
        if p and p[0].text_lines:
            return " ".join(l.text for l in p[0].text_lines)
        return ""
    except Exception as e:
        return f"[Surya error: {e}]"


def clean_amount_words(text):
    text = " ".join(text.split())

    text = text.replace("lakhh", "lakh")
    text = text.replace("thausand", "thousand")
    text = text.replace("thausnd", "thousand")
    text = text.replace("oniy", "only")
    text = text.replace("onlv", "only")

    return text


# --- Date specialist -------------------------------------------------------
# The only fine-tuned model worth keeping: the synthetic no-slash Date model,
# which fixed base TrOCR's date hallucination. All other fields use base TrOCR,
# which beats every fine-tuned variant on Name / Amount_Words / Amount_Number.
# Path comes from config.py (override via the TROCR_FIELD_MODELS_DIR env var).
from config import DATE_MODEL_DIR as DATE_DIR
date_p = None
date_m = None


def _ft_preprocess(img):
    # The Date specialist was trained on BINARIZED crops, so match that here
    # (this is the old Otsu-threshold preprocess, not the base-TrOCR one).
    ia = np.asarray(img)
    if ia.ndim == 3 and ia.shape[2] == 3:
        g = cv2.cvtColor(ia, cv2.COLOR_RGB2GRAY)
    elif ia.ndim == 3 and ia.shape[2] == 4:
        g = cv2.cvtColor(ia, cv2.COLOR_RGBA2GRAY)
    else:
        g = ia
    d = cv2.fastNlMeansDenoising(g, None, 10, 7, 21)
    _, t = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.copyMakeBorder(t, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(cv2.cvtColor(b, cv2.COLOR_GRAY2RGB))


def _run_date_model(i):
    global date_p, date_m
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    if date_m is None:
        date_p = TrOCRProcessor.from_pretrained(DATE_DIR)
        date_m = VisionEncoderDecoderModel.from_pretrained(DATE_DIR).to(DEVICE).eval()
    px = date_p(images=_ft_preprocess(i), return_tensors="pt").pixel_values.to(DEVICE)
    ids = date_m.generate(px, max_new_tokens=64, num_beams=4,
                          no_repeat_ngram_size=3, repetition_penalty=1.8,
                          early_stopping=True)
    return date_p.batch_decode(ids, skip_special_tokens=True)[0].strip()


def run_trocr(i):
    global trocr_p, trocr_m

    try:
        if i is None or (
            hasattr(i, "size") and i.size == 0
        ):
            return ""

        # Date is the one field where the synthetic no-slash model beats base.
        if _current_field == "Date":
            return _run_date_model(i)

        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel
        )

        if trocr_p is None:
            trocr_p = TrOCRProcessor.from_pretrained(
                TROCR_MODEL
            )

            trocr_m = VisionEncoderDecoderModel.from_pretrained(
                TROCR_MODEL
            ).to(DEVICE)

            trocr_m.eval()

        pi = _trocr_preprocess(i)

        pixel_values = trocr_p(
            images=pi,
            return_tensors="pt"
        ).pixel_values.to(DEVICE)

        generated_ids = trocr_m.generate(
            pixel_values,

            # Cheque amounts can be long.
            max_new_tokens=128,

            # Simpler decoding.
            num_beams=1,

            do_sample=False,

            early_stopping=True
        )

        text = trocr_p.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        text = clean_amount_words(text.strip())

        return text

    except Exception as e:
        return f"[TrOCR error: {e}]"


def run_rapidocr(i):
    global rapid
    try:
        if i is None:
            return ""
        from rapidocr_onnxruntime import RapidOCR
        if rapid is None:
            rapid = RapidOCR()
        r, _ = rapid(np.asarray(i))
        if r:
            t = " ".join(x[1].strip() for x in r)
            return t if t not in ["-", "_", "___"] else ""
        return ""
    except Exception as e:
        return f"[RapidOCR error: {e}]"


# --- Numeric cleanup --------------------------------------------------------
# OCR commonly misreads digits in amount / account-number fields (O for 0, S for
# 5, and so on). We only correct tokens that already look numeric, so ordinary
# words - names included - are left exactly as the engine produced them. Nothing
# is hard-coded to a particular cheque, so this works on any input.

_CONF = {
    'O': '0', 'o': '0', 'D': '0', 'Q': '0',
    'l': '1', 'I': '1', 'i': '1', '|': '1',
    'Z': '2', 'z': '2', 'S': '5', 's': '5',
    'b': '6', 'G': '6', 'B': '8',
}


def _looks_numeric(tok):
    # Only touch tokens that contain at least one real digit and are mostly
    # made of digits, separators, or known digit look-alikes.
    if not any(c.isdigit() for c in tok):
        return False
    relevant = sum(c.isdigit() or c in ',./-' or c in _CONF for c in tok)
    return relevant / len(tok) >= 0.6


def _collapse_tokens(toks, max_rep=2, max_k=4):
    out = []
    i, n = 0, len(toks)
    while i < n:
        hit = False
        for k in range(1, max_k + 1):
            if i + k > n:
                break
            unit = toks[i:i + k]
            reps, j = 1, i + k
            while j + k <= n and toks[j:j + k] == unit:
                reps += 1
                j += k
            if reps > max_rep:
                out.extend(unit * max_rep)
                i = j
                hit = True
                break
        if not hit:
            out.append(toks[i])
            i += 1
    return out


def collapse_repeats(text):
    # OCR decoders can loop on low-contrast strips, repeating a short phrase many
    # times. Collapse any 1-4 word phrase repeated more than twice in a row.
    return "\n".join(" ".join(_collapse_tokens(line.split())) for line in text.split("\n"))


def fix_numeric(text):
    out = []
    for line in text.split('\n'):
        toks = []
        for t in line.split():
            if _looks_numeric(t):
                t = ''.join(_CONF.get(c, c) for c in t)
            toks.append(t)
        out.append(' '.join(toks))
    return '\n'.join(out)


# --- Amount-in-words vocabulary correction ---------------------------------
# Snap each word in the Amount_Words field to the closest term in the reference
# vocabulary below. Spelling is corrected (e.g. a misread "crove" -> "crore"),
# but the original capitalisation from the cheque and any surrounding
# punctuation are preserved. Tokens with no close match are left untouched, so
# unexpected content is never silently discarded.

_AMOUNT_WORDS_VOCAB = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
    "seventy", "eighty", "ninety", "hundred", "hundreds", "thousand", "thousands",
    "lakh", "lakhs", "lack", "lacks", "crore", "only", "and",
]
_AMOUNT_WORDS_SET = set(_AMOUNT_WORDS_VOCAB)


def _lev_str(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _match_case(src, word):
    # Apply the original token's capitalisation to the corrected word.
    if src.isupper():
        return word.upper()
    if src[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def correct_amount_words(text):
    out_lines = []
    for line in text.split("\n"):
        toks = []
        for tok in line.split():
            # Peel off leading / trailing punctuation so it can be reattached.
            lead, trail, core = "", "", tok
            while core and not core[0].isalnum():
                lead += core[0]
                core = core[1:]
            while core and not core[-1].isalnum():
                trail = core[-1] + trail
                core = core[:-1]

            if core.isalpha() and len(core) >= 2:
                low = core.lower()
                if low in _AMOUNT_WORDS_SET:
                    best = low
                else:
                    # Allow more edits for longer words; short words must match
                    # almost exactly to avoid over-correcting real words.
                    allowed = 1 if len(low) <= 4 else (2 if len(low) <= 8 else 3)
                    best, bestd = low, allowed + 1
                    for w in _AMOUNT_WORDS_VOCAB:
                        if abs(len(w) - len(low)) > allowed:
                            continue
                        dd = _lev_str(low, w)
                        if dd < bestd:
                            bestd, best = dd, w
                    if bestd > allowed:
                        best = low  # no close match -> keep original
                toks.append(lead + _match_case(core, best) + trail)
            else:
                toks.append(tok)
        out_lines.append(" ".join(toks))
    return "\n".join(out_lines)


o_map = {
    "EasyOCR": run_easyocr,
    "Surya": run_surya,
    "TrOCR": run_trocr,
    "RapidOCR": run_rapidocr,
}


# --- Layout helpers ---------------------------------------------------------
# Crops arrive as (image, x1, y1). Thresholds are derived from the typical box
# height so grouping behaves the same at any scan resolution.

def _hw(img):
    if isinstance(img, Image.Image):
        w, h = img.size
        return h, w
    a = np.asarray(img)
    return a.shape[0], a.shape[1]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _dedupe(crops, thr=0.6):
    # Drop near-duplicate detections - the same field boxed more than once,
    # which happens when the detector runs at a low confidence threshold.
    boxed = []
    for c in crops:
        h, w = _hw(c[0])
        boxed.append((c, (c[1], c[2], c[1] + w, c[2] + h), w * h))
    boxed.sort(key=lambda e: e[2], reverse=True)  # largest box first
    kept = []
    for c, box, _ in boxed:
        if all(_iou(box, kb) <= thr for _, kb in kept):
            kept.append((c, box))
    return [c for c, _ in kept]


def _group_lines(crops):
    hs = [_hw(c[0])[0] for c in crops]
    mh = float(np.median(hs)) if hs else 40.0
    row_tol = mh * 0.8      # tops within this distance -> same row
    word_gap = mh * 2.0     # horizontal gap larger than this -> new line segment

    # Cluster into rows by vertical position (top to bottom).
    rows = []
    cur = []
    for c in sorted(crops, key=lambda c: c[2]):
        if cur and abs(c[2] - cur[-1][2]) >= row_tol:
            cur.sort(key=lambda x: x[1])
            rows.append(cur)
            cur = []
        cur.append(c)
    if cur:
        cur.sort(key=lambda x: x[1])
        rows.append(cur)

    # Split each row wherever the horizontal gap is large.
    lines = []
    for row in rows:
        seg = [row[0]]
        for i in range(1, len(row)):
            prev, c = row[i - 1], row[i]
            pw = _hw(prev[0])[1]
            gap = c[1] - (prev[1] + pw)
            if gap > word_gap:
                lines.append(seg)
                seg = [c]
            else:
                seg.append(c)
        lines.append(seg)
    return lines, mh


def _stitch(line, mh):
    # Join a line's crops into one image with white spacers, for engines that
    # read a whole line at once (Surya, TrOCR).
    spacer = max(8, int(mh * 0.3))
    imgs = []
    for it in line:
        ia = np.asarray(it[0])
        if ia.ndim == 2:
            ia = cv2.cvtColor(ia, cv2.COLOR_GRAY2RGB)
        elif ia.shape[2] == 4:
            ia = cv2.cvtColor(ia, cv2.COLOR_RGBA2RGB)
        imgs.append(ia)
    if not imgs:
        return None
    th = max(im.shape[0] for im in imgs)
    parts = []
    for im in imgs:
        h, w = im.shape[:2]
        if h != th:
            scale = th / float(h)
            im = cv2.resize(
                im,
                (
                    int(w * scale),
                    th
                ),
                interpolation=cv2.INTER_CUBIC
            )
        parts.append(im)
        parts.append(np.full((th, spacer, 3), 255, dtype=np.uint8))
    parts.pop()
    return np.hstack(parts)


# The order fields are reported in. Only the values are printed - one field per
# line, in this sequence - not the label names themselves.
FIELD_ORDER = ["Date", "Name", "Amount_Words", "Amount_Number"]


def _label(c):
    # Field name if the detector supplied one, else None.
    return c[3] if len(c) > 3 and c[3] else None


def _run_engine(crops, engine, on):
    # Read a set of crops with one engine, in reading order, and return cleaned text.
    global _current_field
    _current_field = _label(crops[0]) if crops else None
    lines, mh = _group_lines(crops)
    out = []
    if on in ("Surya", "TrOCR"):
        for line in lines:
            sim = _stitch(line, mh)
            if sim is not None:
                t = engine(sim)
                if t:
                    out.append(t)
    else:
        for line in lines:
            parts = [p for p in (engine(it[0]) for it in line) if p]
            if parts:
                out.append(" ".join(parts))
    result = fix_numeric(collapse_repeats("\n".join(out).strip()))

    # Snap amount-in-words output to the reference vocabulary (case preserved).
    if _current_field == "Amount_Words":
        result = correct_amount_words(result)

    return result


def extract_text(ps, pic, sel):
    if not ps or not pic:
        return gr.update(value="No detection images.", visible=True)

    res = []
    for x in ps:
        m = x.replace("Output: ", "").strip()
        if m not in pic:
            continue
        crps = pic[m].get('c', [])
        res.append(f"====== {m} Detection ======\n")

        valid = [c for c in crps
                 if isinstance(c, (list, tuple)) and len(c) >= 3
                 and isinstance(c[0], (np.ndarray, Image.Image))]
        if not valid:
            res.append("(no usable crops)\n\n")
            continue

        valid = _dedupe(valid)
        labeled = any(_label(c) for c in valid)

        for on in sel:
            engine = o_map.get(on, run_rapidocr)
            res.append(f"--- OCR: {on} ---\n")

            if labeled:
                # Known fields, in the fixed sequence (values only, no labels).
                for field in FIELD_ORDER:
                    fc = [c for c in valid if _label(c) == field]
                    if fc:
                        res.append(f"{_run_engine(fc, engine, on)}\n")
                # Any labels outside the known set.
                extra = [c for c in valid if _label(c) and _label(c) not in FIELD_ORDER]
                if extra:
                    res.append(f"{_run_engine(extra, engine, on)}\n")
                # Crops with no label fall back to reading order.
                unl = [c for c in valid if not _label(c)]
                if unl:
                    res.append(f"{_run_engine(unl, engine, on)}\n")
            else:
                # No labels at all: top-to-bottom reading order.
                res.append(f"{_run_engine(valid, engine, on)}\n")

            res.append("\n")

    return gr.update(value="".join(res), visible=True)
