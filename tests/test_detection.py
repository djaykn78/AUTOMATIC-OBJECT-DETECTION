import sys, json, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detection.detection import detect_objects, draw_detections

logging.basicConfig(level=logging.INFO)

img_path = sys.argv[1] if len(sys.argv) > 1 else "sample_images/1.jpeg"
backend = sys.argv[2] if len(sys.argv) > 2 else "dino"
thr = float(sys.argv[3]) if len(sys.argv) > 3 else None

objs = detect_objects(img_path, confidence_threshold=thr, backend=backend, fallback=False)
for o in objs:
    print(json.dumps(o))
print(f"{len(objs)} objects [{backend}]")
Path("output").mkdir(exist_ok=True)
draw_detections(img_path, objs, f"output/debug_{backend}.jpg")