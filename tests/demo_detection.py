import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget

from app.detection.detection import detect_objects, draw_detections


class Worker(QThread):
    done = Signal(str, object)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        self.done.emit(self.path, detect_objects(self.path))


class Demo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WallMorph - Detection Demo")
        self.resize(1000, 700)
        self.button = QPushButton("Open image")
        self.status = QLabel("Choose an image. The first run also loads the model, so it takes longer.")
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(self.button)
        layout.addWidget(self.status)
        layout.addWidget(self.view, 1)
        self.button.clicked.connect(self.pick)
        self.worker = None

    def pick(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "sample_images",
                                              "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path:
            return
        self.button.setEnabled(False)
        self.status.setText("Detecting...")
        self.worker = Worker(path)
        self.worker.done.connect(self.show_result)
        self.worker.start()

    def show_result(self, path, objects):
        img = draw_detections(path, objects)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.view.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.view.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        counts = dict(Counter(o["type"] for o in objects))
        self.status.setText(f"{len(objects)} objects: {counts}" if objects
                            else "No objects found (manual selection would take over here)")
        self.button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec())