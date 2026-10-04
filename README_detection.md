# WallMorph: Automatic Object Detection Module

Module owner: Person 3 (ML / Object Detection)
File: `app/detection/detection.py`

This module takes a photo of a room or wall and returns the doors, windows, wardrobes, switches, lights and other objects it finds, each with a position, size and confidence score. The rest of the team can use it without knowing anything about the model inside.

```python
from app.detection.detection import detect_objects

objects = detect_objects("sample_images/1.jpeg")
for obj in objects:
    print(obj["type"], obj["x"], obj["y"], obj["width"], obj["height"], obj["confidence"])
```

---

## 1. Quick summary

| Question | Answer |
|---|---|
| What model does it use? | **Grounding DINO (base)**, an open-vocabulary detector driven by text prompts |
| Why not plain YOLO? | The standard YOLO (COCO) classes have no window, door, switchboard or wardrobe. It returned nothing on 11 of our 13 test images |
| Does it need a GPU? | No. CPU works (slower). A GPU is optional |
| Speed | About 7-8 s per image on a 4 GB NVIDIA GPU. The first call also loads about 1 GB of weights |
| What if it fails? | `detect_objects` never raises. It returns `[]`, and the UI should fall back to manual selection |
| Accuracy on our 13 images | 10 images fully correct, 3 with a known limit (see section 8) |

---

## 2. Project folder layout

```
Mini Project/
|-- app/
|   `-- detection/
|       |-- __init__.py          (empty file, needed for imports)
|       `-- detection.py         (the module)
|-- tests/
|   |-- test_detection.py        (run one image, one backend)
|   |-- test_batch.py            (run every image in sample_images, save comparison images)
|   `-- debug_raw.py             (print raw model scores for one image)
|-- sample_images/               (13 test photos: 1.jpeg ... 13.jpeg)
|-- output/                      (debug images are written here, ignored by git)
|-- requirements-cpu.txt
|-- requirements-gpu.txt
`-- README_detection.md
```

---

## 3. Setup

### Windows, with an NVIDIA GPU

Run `WallMorph_Setup_GPU.bat`, then activate the environment and add one package:

```powershell
venv_gpu\Scripts\activate
pip install transformers
```

Add `transformers` as a line in `requirements-gpu.txt` (tell Person 6 first, as the team rules require).

### CPU only (any machine)

```powershell
python -m venv venv_cpu
venv_cpu\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
pip install transformers
```

### Check that the GPU is used

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`True` means GPU, `False` means CPU. The code picks the right one automatically.

### Model downloads

Weights download automatically on the first run and are cached:

| Model | Size | Used for |
|---|---|---|
| `IDEA-Research/grounding-dino-base` | about 1 GB | default backend `dino` |
| `IDEA-Research/grounding-dino-tiny` | about 700 MB | faster backend `dino_tiny` |
| `yolov8s-worldv2.pt` | 25 MB | fallback `yoloworld` |
| `yolov8m.pt` | 50 MB | fallback `coco` |

---

## 4. How to use it

### Main function

```python
detect_objects(image, confidence_threshold=None, backend="dino",
               min_area_ratio=0.0005, iou_threshold=0.5, fallback=True)
```

| Argument | Meaning |
|---|---|
| `image` | File path, OpenCV array (BGR) or PIL image |
| `confidence_threshold` | `None` (recommended) uses the tuned per-type thresholds. A number, such as `0.4`, applies one threshold to everything |
| `backend` | `"dino"` (default), `"dino_tiny"`, `"dino_base"`, `"yoloworld"`, `"ensemble"`, `"coco"` |
| `fallback` | `True`: if the backend fails, try the next one. `False`: use only this backend (for testing) |

### Helper

```python
draw_detections(image, objects, save_path="output/debug.jpg")   # boxes + labels, for debugging and the demo
```

### Output format (the team-wide common object format)

```python
{
    "id": 1,
    "type": "window",
    "x": 635, "y": 253, "width": 433, "height": 420,   # x, y = top-left corner, in pixels
    "rotation": 0,
    "scale": 1.0,
    "design": None,
    "confidence": 0.77,
    "source": "automatic"
}
```

`type` is one of: `window`, `door`, `wardrobe`, `bookshelf`, `table`, `light`, `switchboard`, `decoration`, `curtain`, `bed`, `whiteboard`.

### Rules for the UI developer (Person 6)

1. Call `detect_objects` once on a blank array at app start to load the model (`np.zeros((64, 64, 3), np.uint8)`).
2. Detection takes several seconds, so run it in a worker thread (for example `QThread` in PySide6) so the window does not freeze.
3. An empty list `[]` is a valid result. Show the manual selection tool.
4. Some labels can be wrong (section 8), so the manual module should let users **change the type** of a detected object and **add** missed ones.

---

## 5. How it works

```
Image
  |
  v
Grounding DINO, 4 passes (one per prompt group)
  |   group 1: door, window, curtain
  |   group 2: wardrobe, cupboard, cabinet, bookshelf, wall shelf, table, desk, bed, cot
  |   group 3: wall light, ceiling light, tube light, chandelier, picture frame,
  |            wall painting, wall clock, poster, whiteboard, blackboard
  |   group 4: light switch, switchboard, electrical socket
  v
Raw boxes [(type, x1, y1, x2, y2, confidence)]
  |
  v
Filters
  |-- drop boxes that are too big or too small for their type
  |-- drop switches with an impossible width/height ratio (door handles)
  |-- drop very weak boxes (relaxed floor)
  v
Overlap removal (NMS)
  |-- the best label wins when boxes cover the same region
  |-- if the winner fails its per-type minimum, it is removed and the runner-up gets a chance
  |-- tighter box replaces a looser one for lights, switches, decorations (removes shadow/glow boxes)
  v
Remove lights/switches that sit inside a door, window, wardrobe, bed ...
  v
Common object format
```

### Why open-vocabulary detection?

YOLO is trained on a fixed list of 80 classes (COCO). Grounding DINO takes text prompts, such as "window" or "light switch", and finds matching regions in the image, so the object list can change by editing a Python dictionary instead of training a model.

### Why prompt groups?

- Putting all prompts in one pass was less accurate, because phrases compete for the same text tokens.
- Putting each type in its own pass (12 passes) was too slow, and a lone "window" prompt matched every rectangular frame.
- **Four groups** is the compromise. Confusable types (door/window) share a pass so the model must choose between them. Switches get their own pass so lights do not dilute their scores.

---

## 6. Final configuration

All of these are constants at the top of `detection.py`. Tune them there.

### Per-type minimum confidence (used when `confidence_threshold=None`)

| Type | Min | Type | Min |
|---|---|---|---|
| window | 0.35 | light | 0.35 |
| door | 0.30 | switchboard | 0.28 |
| wardrobe | 0.30 | decoration | 0.30 |
| bookshelf | 0.45 | curtain | 0.50 |
| table | 0.36 | bed | 0.40 |
| whiteboard | 0.40 | | |

The model itself is asked for boxes at 0.25 and above. `dino_tiny` and `yoloworld` require +0.10 on top, because they produce more false boxes.

### Per-type box size limits (fraction of image area)

Examples: a switchboard must cover 0.03% to 3% of the image, a door 2% to 60%, a light 0.03% to 8%. The full table is `TYPE_AREA` in the file. A switch must also have a width/height ratio between 0.35 and 6 (`TYPE_ASPECT`), which removes thin door handles.

---

## 7. What we tried (experiment log)

All tests used the 13 photos in `sample_images/`, run with `tests/test_batch.py`.

### Experiment 1: which model family?

| Model | Result |
|---|---|
| **COCO YOLO** (`yolov8m`) | Returned nothing on 11 of 13 images. The 2 non-empty results were wrong labels. No window/door/switch/wardrobe classes exist |
| **YOLO-World** (`yolov8s-worldv2`) | Fast (under 1 s) but unreliable. 0 objects on 5 of 13 images at default threshold. Labeled a door as "bookshelf". Missed windows |
| **Grounding DINO tiny** | Door 0.91 and window 0.89 on our first image. Slower (about 8-10 s) but far better |
| **Ensemble** (YOLO-World + DINO) | No improvement over DINO alone. It only added YOLO-World's wrong labels |

Decision: **Grounding DINO**.

### Experiment 2: threshold tuning

| Setting | Result |
|---|---|
| Global threshold 0.20 | Mostly noise: 16-20 boxes per image, for example 5 doors on one image |
| Global threshold 0.30 / 0.40 | Misses small switches and correct doors scoring 0.30-0.38 |
| **Per-type minimums + size limits** | Removed whole-image boxes, bottles labeled as lights, handles labeled as switches |

Decision: **per-type thresholds**, not one global number.

### Experiment 3: tiny vs base model

| | `dino_tiny` | `dino_base` (chosen) |
|---|---|---|
| Speed | 8-10 s | 11-14 s (7-9 s after the speed fix) |
| Small switches | Found more | Missed some at first (fixed by the separate switch pass) |
| False boxes | Many (curtain on a chair, latches as switches, jar as light) | Few |
| Chair-only photo (should be empty) | 2 false boxes | 0 |

Decision: **base** as default, **tiny** as the low-resource option.

### Experiment 4: speed and GPU memory

- Running all 12 prompt types in one batch needed about 3.8 GB, ran out of memory on a 4 GB GPU, and spilled into system RAM. Each image took **100-200 s**.
- Sequential single-type passes: **34-41 s** per image.
- Four prompt groups run sequentially: **7-9 s** per image.

### Experiment 5: post-processing bugs and fixes

| Problem seen | Fix |
|---|---|
| Wall lamp labeled `switchboard` instead of `light` | Resolve overlaps first, then apply the strict per-type minimum |
| Kitchen window disappeared (the correct label scored 0.358, below a 0.40 minimum) | Raw scores checked with `debug_raw.py`. `window` minimum lowered to 0.35, and failed winners are removed so the runner-up can compete |
| Three labels on one door (door, window, switchboard) | Nested boxes of similar size and big types are merged |
| Lamp box included its shadow | For lights, switches, decorations: the tighter box replaces the looser one |
| Door handles and glass labeled as switches/lights | Drop fixtures inside doors/windows/wardrobes, and check aspect ratio |
| Chair-only photo produced `window` / `curtain` / `wardrobe` | Per-type minimums and a bigger confidence bar for the tiny model |
| `UnboundLocalError: pool` | Indentation mistake in the retry loop |
| Backend silently fell back to YOLO-World and looked like a DINO result | Added `fallback=False` for tests and logged every fallback |

---

## 8. Final results and known limits

### Final batch result (13 images)

| Image | Result |
|---|---|
| 1 | Door, window, 2 switch plates: correct |
| 3, 4, 11 | Wardrobes, doors, desk, sockets: correct |
| 6, 7 | Lamp and switches: correct |
| 8 | Door with side windows: window frame + door |
| 9 | Kitchen window: correct |
| 10 | Cove light, window, switches: correct |
| 12 | Chair only: **0 detections** (correct) |
| 2 | Both doors and the socket found. **Missed** the photo frames, floating shelves and the left-wall switch |
| 5 | The narrow window is labeled `door` (62%) |
| 13 | The doorway at the left edge is labeled `window` |

### Known limitations

1. **Door/window labels can swap.** The box is right, the label is wrong. The manual module should offer a "change type" dropdown.
2. **Small frames and shelves can be missed.** The manual module should offer "add object".
3. **No class exists** for the ceiling cove glow or an open doorway.
4. **Some wardrobes get two boxes** (body and top cabinets).
5. **CPU speed with `dino_base` has not been measured** on a CPU-only machine. If it is too slow, use `backend="dino_tiny"`.

---

## 9. How to run the tests

All commands are run from the project root.

```powershell
# One image, one backend (saves output\debug_<backend>.jpg)
python tests\test_detection.py sample_images\1.jpeg dino

# Optional threshold override
python tests\test_detection.py sample_images\1.jpeg dino 0.3

# Every image in sample_images, with comparison images in output\batch\<image>\_compare.jpg
python tests\test_batch.py

# Raw model scores for one image (use this when a box is missing)
python tests\debug_raw.py sample_images\9.jpeg
```

To compare configurations in `test_batch.py`, edit the `CONFIGS` list. Each entry is `(label, backend, threshold)`:

```python
CONFIGS = [
    ("base_v3", "dino", None),
    ("tiny",    "dino_tiny", None),
]
```

Output files: `output/batch/summary.csv`, `output/batch/results.json`, and the debug images.

---

## 10. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `No module named 'transformers'` | `pip install transformers`. Without it, the code falls back to YOLO-World, and results look much worse |
| `CUDA out of memory` | Close other GPU apps, or use `backend="dino_tiny"`, or run on CPU. The current code runs passes sequentially to avoid this |
| Everything takes minutes | The GPU is not being used (check `torch.cuda.is_available()`), or memory is spilling into system RAM |
| `huggingface_hub ... symlinks` warning on Windows | Harmless. Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to hide it |
| `FutureWarning: labels will return integer ids` | Harmless. The code handles it |
| Result is `[]` | Check the log for "Image error" or "Backend ... failed". `[]` is also valid for an image with no known objects |
| A real object is missing | Run `debug_raw.py` on that image. If the score is just below the type's minimum, lower it in `TYPE_MIN_CONF` and recheck the chair image (12) for false boxes |
| Wrong label on a good box | Known limitation. Fix it in the UI by changing the type |

---

## 11. How to change what it detects

**Add or rename an object type**

1. Add it to `PROMPTS` (type, list of prompt phrases).
2. Put the same phrases into one of the lists in `DINO_GROUPS`.
3. Add entries to `TYPE_MIN_CONF` and `TYPE_AREA`.
4. Run `test_batch.py` and look at the debug images.

Tips: specific phrases work better than generic ones ("wall light" is better than "lamp", because "lamp" matched steel bottles). Check the chair photo (12) after any change, since it should stay empty.

---

## 12. Module contract (for the team)

```
MODULE:       Object Detection
FILE:         app/detection/detection.py
INPUT:        image path, numpy array (BGR) or PIL image
OUTPUT:       list of common-format dicts (source="automatic"); [] on failure
FUNCTIONS:    detect_objects(image, confidence_threshold=None, backend="dino", fallback=True)
              draw_detections(image, objects, save_path)
DEPENDENCIES: torch, transformers, opencv-python, numpy, pillow
              (ultralytics only for the YOLO-World / COCO fallbacks)
HOW TO RUN:   python tests\test_detection.py sample_images\1.jpeg
LIMITATIONS:  door/window labels can swap; small decorations and shelves can be missed;
              the manual selection module is the fallback
```

---

## 13. Viva cheat sheet

**What is Grounding DINO?** An open-vocabulary detector. Given an image and a text prompt such as "window", it predicts boxes and a confidence score for matching regions.

**What is a bounding box?** The rectangle around a detected object. The model returns corners (x1, y1, x2, y2). We convert them to x, y, width, height with `width = x2 - x1` and `height = y2 - y1`.

**What is confidence?** How strongly the model believes the box matches the label. Open-vocabulary scores are lower than classic YOLO scores. A correct window may score only 0.35 to 0.8, so a 0.5 cut-off would remove most real objects.

**Why not just use pretrained YOLO?** It has no class for windows, doors, switchboards or wardrobes. It returned nothing on 11 of 13 of our photos.

**Why not train a custom model?** We had days, not weeks, and no labelled dataset. Open-vocabulary detection gives usable results with no training.

**What is NMS?** Non-maximum suppression. When several boxes cover the same object, it keeps the highest-confidence one and removes the rest.

**What if the model misses something or labels it wrongly?** The user can add or correct it with the manual selection module (Person 5).

**What could be improved?**
- Fine-tune a detector on labelled switchboards and frames.
- Use SAM (Segment Anything) to turn boxes into pixel-accurate masks for wallpaper and texture replacement.
- Add a decoration-only pass with `dino_tiny` to recover small frames.
