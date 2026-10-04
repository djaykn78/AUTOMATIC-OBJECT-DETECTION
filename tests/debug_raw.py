import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detection.detection import BACKENDS, load_image

img = load_image(sys.argv[1])
for t, x1, y1, x2, y2, c in sorted(BACKENDS["dino"](img, 0.25), key=lambda d: -d[5]):
    print(f"{t:12s} {c:.3f}  ({int(x1)},{int(y1)},{int(x2)},{int(y2)})")