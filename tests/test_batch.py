import sys, json, csv, time, logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detection.detection import detect_objects, draw_detections, load_image

logging.basicConfig(level=logging.WARNING)
for n in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.ERROR)

IMG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample_images")
OUT = Path("output/batch")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# (label, backend, confidence_threshold)  None = backend default
CONFIGS = [
    ("base_v3", "dino", None),
]

COLS = 3
TILE_W = 640


def montage(tiles):
    """tiles: list of (title, annotated BGR image) -> single grid image."""
    cells = []
    for title, im in tiles:
        s = TILE_W / im.shape[1]
        im = cv2.resize(im, (TILE_W, int(im.shape[0] * s)))
        bar = np.full((34, TILE_W, 3), 30, np.uint8)
        cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cells.append(np.vstack([bar, im]))
    h = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for c in cells]
    while len(cells) % COLS:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + COLS]) for i in range(0, len(cells), COLS)]
    return np.vstack(rows)


def main():
    images = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        print(f"No images found in {IMG_DIR.resolve()}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(images)} images x {len(CONFIGS)} configs\n")

    # warm-up so model loading is not counted in timings
    blank = np.zeros((64, 64, 3), np.uint8)
    for b in sorted({c[1] for c in CONFIGS}):
        print(f"loading {b} ...")
        detect_objects(blank, backend=b, fallback=False)
    print()

    results, rows = {}, []
    for p in images:
        img = load_image(p)
        d = OUT / p.stem
        d.mkdir(exist_ok=True)
        tiles = []
        print(f"=== {p.name} ({img.shape[1]}x{img.shape[0]}) ===")
        results[p.name] = {}
        for label, backend, thr in CONFIGS:
            t0 = time.time()
            objs = detect_objects(img, confidence_threshold=thr, backend=backend, fallback=False)
            dt = time.time() - t0
            counts = dict(Counter(o["type"] for o in objs))
            annotated = draw_detections(img, objs, d / f"{label}.jpg")
            tiles.append((f"{label}  n={len(objs)}  {dt:.1f}s", annotated))
            results[p.name][label] = objs
            rows.append([p.name, label, len(objs), json.dumps(counts), round(dt, 2)])
            print(f"  {label:<18} n={len(objs):<3} {dt:5.1f}s  {counts}")
        cv2.imwrite(str(d / "_compare.jpg"), montage(tiles))
        print()

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "config", "objects", "types", "seconds"])
        w.writerows(rows)
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)

    print(f"Saved to {OUT.resolve()}")
    print("  <image>/_compare.jpg   all configs side by side")
    print("  <image>/<config>.jpg   individual debug images")
    print("  summary.csv, results.json")


if __name__ == "__main__":
    main()