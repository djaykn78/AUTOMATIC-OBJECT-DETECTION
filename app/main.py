"""WallMorph application entry point."""

import sys
from PySide6.QtWidgets import QApplication, QLabel

def main() -> int:
    app = QApplication(sys.argv)
    window = QLabel("WallMorph - Interactive Wall Redesigning System")
    window.setWindowTitle("WallMorph")
    window.resize(1000, 650)
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
