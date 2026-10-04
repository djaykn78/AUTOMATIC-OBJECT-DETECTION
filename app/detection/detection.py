"""Automatic object detection for WallMorph.

Main API:
    objects = detect_objects(image, confidence_threshold=None, backend="dino", fallback=True)
Returns a list of dicts in the common object format. Never raises; returns [] on failure.
Backends: "dino" (Grounding DINO base, best), "dino_tiny" (faster, less accurate),
          "yoloworld", "ensemble", "coco".
"""
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("wallmorph.detection")

# project type -> text prompts
PROMPTS = {
    "window": ["window"],
    "door": ["door"],
    "wardrobe": ["wardrobe", "cupboard", "cabinet"],
    "bookshelf": ["bookshelf", "wall shelf"],
    "table": ["table", "desk"],
    "light": ["wall light", "ceiling light", "tube light", "chandelier"],
    "switchboard": ["light switch", "switchboard", "electrical socket"],
    "decoration": ["picture frame", "wall painting", "wall clock", "poster"],
    "curtain": ["curtain"],
    "bed": ["bed", "cot"],
    "whiteboard": ["whiteboard", "blackboard"],
}
PROMPT_TO_TYPE = {p: t for t, ps in PROMPTS.items() for p in ps}
ALL_PROMPTS = list(PROMPT_TO_TYPE)

# Grounding DINO prompt groups (one forward pass each). Confusable types share a pass
# so the model must choose between them; switches get their own pass so lights
# do not dilute their scores.
DINO_GROUPS = [
    ["door", "window", "curtain"],
    ["wardrobe", "cupboard", "cabinet", "bookshelf", "wall shelf", "table", "desk", "bed", "cot"],
    ["wall light", "ceiling light", "tube light", "chandelier",
     "picture frame", "wall painting", "wall clock", "poster", "whiteboard", "blackboard"],
    ["light switch", "switchboard", "electrical socket"],
]

COCO_MAP = {"dining table": "table", "clock": "decoration", "potted plant": "decoration",
            "vase": "decoration", "tv": "decoration", "bed": "bed"}

DINO_TINY = "IDEA-Research/grounding-dino-tiny"
DINO_BASE = "IDEA-Research/grounding-dino-base"
MODEL_NAME = {"yoloworld": "yolov8s-worldv2.pt", "coco": "yolov8m.pt", "dino": DINO_BASE}

# threshold passed to the model itself
DEFAULT_THRESHOLD = {"yoloworld": 0.10, "dino": 0.25, "dino_tiny": 0.25, "dino_base": 0.25,
                     "coco": 0.40, "ensemble": 0.25}

# final per-type minimum confidence (used only when confidence_threshold is None)
TYPE_MIN_CONF = {"window": 0.35, "door": 0.30, "wardrobe": 0.30, "bookshelf": 0.45, "table": 0.36,
                 "light": 0.35, "switchboard": 0.28, "decoration": 0.30, "curtain": 0.50,
                 "bed": 0.40, "whiteboard": 0.40}
# weaker backends must be more confident (tiny produced many false boxes)
CONF_BONUS = {"dino_tiny": 0.10, "yoloworld": 0.10, "coco": 0.0}

# (min, max) box area as a fraction of the image area
TYPE_AREA = {"switchboard": (0.0003, 0.03), "light": (0.0003, 0.08), "decoration": (0.0005, 0.25),
             "window": (0.01, 0.60), "door": (0.02, 0.60), "table": (0.01, 0.40),
             "bookshelf": (0.01, 0.50), "wardrobe": (0.03, 0.70), "curtain": (0.02, 0.50),
             "bed": (0.03, 0.60), "whiteboard": (0.02, 0.60)}
DEFAULT_AREA = (0.0005, 0.60)
# (min, max) width/height ratio; removes thin handles detected as switches
TYPE_ASPECT = {"switchboard": (0.35, 6.0)}

BIG_TYPES = {"door", "window", "wardrobe", "bed", "bookshelf", "curtain"}
TIGHT_TYPES = {"light", "switchboard", "decoration"}   # prefer the tighter box over a looser one
FIXTURES = {"light", "switchboard"}                   # cannot sit inside a big structure
CONTAINERS = {"door", "window", "wardrobe", "bed", "bookshelf", "curtain"}

_cache = {}


def _device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_image(image):
    """Accept path, PIL image or numpy array (BGR, as OpenCV). Returns BGR uint8 array."""
    import cv2
    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Invalid or unsupported image: {p}")
        return img
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.astype(np.uint8)
        raise ValueError("Unsupported array shape")
    try:  # PIL
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Unsupported image type: {type(image)}") from e


# ---------- backends: each returns [(type, x1, y1, x2, y2, conf)] ----------

def _run_yoloworld(img, thr):
    from ultralytics import YOLOWorld
    if "yw" not in _cache:
        m = YOLOWorld(MODEL_NAME["yoloworld"])
        m.set_classes(ALL_PROMPTS)
        _cache["yw"] = m
    r = _cache["yw"].predict(img, conf=thr, device=_device(), verbose=False)[0]
    out = []
    for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        t = PROMPT_TO_TYPE.get(ALL_PROMPTS[int(k)])
        if t:
            out.append((t, x1, y1, x2, y2, c))
    return out


def _run_coco(img, thr):
    from ultralytics import YOLO
    if "coco" not in _cache:
        _cache["coco"] = YOLO(MODEL_NAME["coco"])
    r = _cache["coco"].predict(img, conf=thr, device=_device(), verbose=False)[0]
    out = []
    for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        t = COCO_MAP.get(r.names[int(k)])
        if t:
            out.append((t, x1, y1, x2, y2, c))
    return out


def _dino_post(proc, outputs, input_ids, thr, sizes):
    kw = dict(target_sizes=sizes, text_threshold=0.25)
    try:
        return proc.post_process_grounded_object_detection(outputs, input_ids, threshold=thr, **kw)
    except TypeError:  # older transformers
        return proc.post_process_grounded_object_detection(outputs, input_ids, box_threshold=thr, **kw)


def _label_to_type(label, phrases):
    """Map DINO's returned phrase to a project type (first-mentioned, then longest, phrase wins)."""
    s = str(label).lower()
    hits = [(s.find(p), -len(p), p) for p in phrases if p in s]
    return PROMPT_TO_TYPE[min(hits)[2]] if hits else None


def _run_dino(img, thr, model_name=None):
    """Grounding DINO, one pass per prompt group, sequential so it fits small GPUs (4 GB)."""
    import cv2, torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    name = model_name or MODEL_NAME["dino"]
    key = ("dino", name)
    if key not in _cache:
        dev = _device()
        _cache[key] = (AutoProcessor.from_pretrained(name),
                       AutoModelForZeroShotObjectDetection.from_pretrained(name).to(dev).eval(), dev)
    proc, model, dev = _cache[key]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = []
    for phrases in DINO_GROUPS:
        inputs = proc(images=pil, text=". ".join(phrases) + ".", return_tensors="pt").to(dev)
        with torch.inference_mode():
            outputs = model(**inputs)
        res = _dino_post(proc, outputs, inputs.input_ids, thr, [pil.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        for box, c, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            t = _label_to_type(lab, phrases) if isinstance(lab, str) else None
            if t:
                out.append((t, *box, c))
        del inputs, outputs
        if dev != "cpu":
            torch.cuda.empty_cache()
    return out


def _run_ensemble(img, thr):
    """DINO + YOLO-World pooled; DINO weighted higher so it wins label conflicts."""
    dets, ok = [], False
    for fn, th, wt in ((_run_yoloworld, max(0.10, thr - 0.15), 0.7), (_run_dino, thr, 1.0)):
        try:
            dets += [(t, x1, y1, x2, y2, c * wt) for t, x1, y1, x2, y2, c in fn(img, th)]
            ok = True
        except Exception as e:
            log.warning("Ensemble part %s failed: %s", fn.__name__, e)
    if not ok:
        raise RuntimeError("all ensemble parts failed")
    return dets


BACKENDS = {
    "dino": _run_dino,
    "dino_base": lambda img, thr: _run_dino(img, thr, DINO_BASE),
    "dino_tiny": lambda img, thr: _run_dino(img, thr, DINO_TINY),
    "yoloworld": _run_yoloworld,
    "coco": _run_coco,
    "ensemble": _run_ensemble,
}
FALLBACK_ORDER = ("dino", "dino_tiny", "yoloworld", "coco")


# ---------- post-processing ----------

def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def _contain(a, b):
    """Fraction of the smaller box that lies inside the other."""
    m = min(_area(a), _area(b))
    return _inter(a, b) / m if m > 0 else 0.0


def _nms(dets, iou_thr=0.5):
    """Highest confidence wins, with three refinements:
    - same type: overlapping/nested boxes are duplicates; for small fixtures
      (light, switch, decoration) a tighter box replaces a looser one (drops shadow/glow boxes);
    - different type: boxes covering nearly the same region (IoU > 0.5) are one object;
    - different big types of near-equal size, one inside the other, are one object."""
    keep = []
    for d in sorted(dets, key=lambda d: d[5], reverse=True):
        drop = False
        for idx, k in enumerate(keep):
            kb, db = k[1:5], d[1:5]
            i = _iou(kb, db)
            if k[0] == d[0]:
                cont = _contain(kb, db)
                if i > iou_thr or cont > 0.85:
                    if (k[0] in TIGHT_TYPES and cont > 0.85 and _area(db) < _area(kb)
                            and d[5] >= 0.8 * k[5]):
                        keep[idx] = d
                    drop = True
                    break
            elif i > 0.5:
                drop = True
                break
            elif k[0] in BIG_TYPES and d[0] in BIG_TYPES:
                small, large = min(_area(kb), _area(db)), max(_area(kb), _area(db))
                if large > 0 and small / large > 0.6 and _contain(kb, db) > 0.8:
                    drop = True
                    break
        if not drop:
            keep.append(d)
    return keep


def _drop_fixtures_in_containers(dets):
    """A light/switch lying (>90%) inside a door, window, wardrobe, ... is hardware or glass, not a fixture."""
    big = [d for d in dets if d[0] in CONTAINERS]
    out = []
    for d in dets:
        a = _area(d[1:5])
        if d[0] in FIXTURES and a > 0 and any(_inter(d[1:5], b[1:5]) / a > 0.9 for b in big):
            continue
        out.append(d)
    return out


def detect_objects(image, confidence_threshold=None, backend="dino",
                   min_area_ratio=0.0005, iou_threshold=0.5, fallback=True):
    """Detect wall objects. Returns list of dicts in the common object format ([] on any failure).

    confidence_threshold=None : backend default + per-type minimums (recommended).
    fallback=True : if `backend` fails, try dino, dino_tiny, yoloworld, coco in that order.
    fallback=False: use only `backend` (testing/comparison).
    """
    try:
        img = load_image(image)
    except Exception as e:
        log.error("Image error: %s", e)
        return []
    h, w = img.shape[:2]

    order = [backend] + ([b for b in FALLBACK_ORDER if b != backend] if fallback else [])
    dets, used = None, None
    for b in order:
        try:
            thr = confidence_threshold if confidence_threshold is not None else DEFAULT_THRESHOLD[b]
            dets = BACKENDS[b](img, thr)
            used = b
            break
        except Exception as e:
            if fallback:
                log.warning("Backend '%s' failed (%s); trying next.", b, e)
            else:
                log.error("Backend '%s' failed: %s", b, e)
    if dets is None:
        return []

      # clip to image, then drop implausible sizes and shapes
        # clip to image, then drop implausible sizes and shapes
    clipped = [(t, max(0, x1), max(0, y1), min(w, x2), min(h, y2), c) for t, x1, y1, x2, y2, c in dets]
    use_type_conf = confidence_threshold is None
    bonus = CONF_BONUS.get(used, 0.0) if use_type_conf else 0.0

    def _min_conf(t):
        return TYPE_MIN_CONF.get(t, 0.3) + bonus

    kept = []
    for d in clipped:
        lo, hi = TYPE_AREA.get(d[0], DEFAULT_AREA)
        a = _area(d[1:5]) / (w * h)
        if not (lo <= a <= hi and a >= min_area_ratio):
            continue
        if d[0] in TYPE_ASPECT:
            bw, bh = d[3] - d[1], d[4] - d[2]
            if bh <= 0 or not (TYPE_ASPECT[d[0]][0] <= bw / bh <= TYPE_ASPECT[d[0]][1]):
                continue
        # relaxed floor: very weak boxes cannot take part in overlap removal
        if use_type_conf and d[5] < _min_conf(d[0]) - 0.06:
            continue
        kept.append(d)

    # 1) the best label wins each overlap, 2) then the strict per-type minimum applies;
    #    a winner that fails its minimum is removed and the runner-up competes
    pool = list(kept)
    while True:
        merged = _nms(pool, iou_threshold)
        if not use_type_conf:
            break
        bad = [d for d in merged if d[5] < _min_conf(d[0])]
        if not bad:
            break
        pool = [d for d in pool if d not in bad]
    final = _drop_fixtures_in_containers(merged)

    objects = []
    for t, x1, y1, x2, y2, c in sorted(final, key=lambda d: (d[2], d[1])):
        objects.append({"id": len(objects) + 1, "type": t,
                        "x": int(round(x1)), "y": int(round(y1)),
                        "width": int(round(x2 - x1)), "height": int(round(y2 - y1)),
                        "rotation": 0, "scale": 1.0, "design": None,
                        "confidence": round(float(c), 3), "source": "automatic"})
    return objects


def draw_detections(image, objects, save_path=None):
    """Debug visualization: returns BGR image with boxes + labels."""
    import cv2
    img = load_image(image).copy()
    for o in objects:
        p1, p2 = (o["x"], o["y"]), (o["x"] + o["width"], o["y"] + o["height"])
        cv2.rectangle(img, p1, p2, (0, 200, 0), 2)
        cv2.putText(img, f'{o["type"]} {o["confidence"]:.2f}', (p1[0], max(15, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if save_path:
        cv2.imwrite(str(save_path), img)
    return img