@echo off
setlocal EnableExtensions
title WallMorph - Complete NVIDIA GPU Setup

echo.
echo ==========================================================
echo          WALLMORPH COMPLETE NVIDIA GPU SETUP
echo ==========================================================
echo.

set "ROOT=%~dp0"
cd /d "%ROOT%"

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found in PATH.
    echo Install Python 3.11.x and enable "Add Python to PATH".
    pause
    exit /b 1
)

for /f "tokens=2" %%A in ('python --version 2^>^&1') do set "PYVER=%%A"
echo Python detected: %PYVER%

echo.
echo Checking NVIDIA driver...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: nvidia-smi was not found.
    echo Install a compatible NVIDIA driver before using GPU acceleration.
    echo The setup will continue, but CUDA may not be available.
) else (
    nvidia-smi
)

echo.
echo [1/7] Creating project directories...
for %%D in (
    app
    app\ui
    app\detection
    app\image_processing
    app\objects
    app\selection
    app\history
    app\database
    assets
    assets\doors
    assets\windows
    assets\bookshelves
    assets\tables
    assets\lights
    assets\switchboards
    assets\decorations
    assets\wallpapers
    assets\textures
    models
    datasets
    projects
    output
    tests
) do if not exist "%%D" mkdir "%%D"

echo.
echo [2/7] Creating project files...

if not exist ".gitignore" (
    >.gitignore echo # Python
    >>.gitignore echo __pycache__/
    >>.gitignore echo *.py[cod]
    >>.gitignore echo *$py.class
    >>.gitignore echo.
    >>.gitignore echo # Virtual environments
    >>.gitignore echo venv/
    >>.gitignore echo .venv/
    >>.gitignore echo venv_cpu/
    >>.gitignore echo venv_gpu/
    >>.gitignore echo.
    >>.gitignore echo # IDE
    >>.gitignore echo .vscode/
    >>.gitignore echo .idea/
    >>.gitignore echo *.code-workspace
    >>.gitignore echo.
    >>.gitignore echo # Secrets
    >>.gitignore echo .env
    >>.gitignore echo .env.*
    >>.gitignore echo.
    >>.gitignore echo # Generated files
    >>.gitignore echo output/
    >>.gitignore echo projects/
    >>.gitignore echo runs/
    >>.gitignore echo.
    >>.gitignore echo # Large ML files
    >>.gitignore echo datasets/
    >>.gitignore echo *.pt
    >>.gitignore echo *.onnx
    >>.gitignore echo *.engine
    >>.gitignore echo *.weights
    >>.gitignore echo.
    >>.gitignore echo # Logs/temp
    >>.gitignore echo *.log
    >>.gitignore echo *.tmp
    >>.gitignore echo *.bak
    >>.gitignore echo.
    >>.gitignore echo # OS
    >>.gitignore echo Thumbs.db
    >>.gitignore echo .DS_Store
)

if not exist "README.md" (
    >README.md echo # WallMorph
    >>README.md echo.
    >>README.md echo Interactive 2D wall visualization and redesigning system.
    >>README.md echo.
    >>README.md echo ## Setup
    >>README.md echo.
    >>README.md echo Run WallMorph_Setup_CPU.bat for CPU setup.
    >>README.md echo Run WallMorph_Setup_GPU.bat for NVIDIA GPU setup.
    >>README.md echo.
    >>README.md echo ## Run
    >>README.md echo.
    >>README.md echo python app\main.py
)

if not exist "requirements-cpu.txt" (
    >requirements-cpu.txt echo numpy==2.2.6
    >>requirements-cpu.txt echo opencv-python==4.10.0.84
    >>requirements-cpu.txt echo Pillow==11.1.0
    >>requirements-cpu.txt echo PySide6==6.8.2
    >>requirements-cpu.txt echo matplotlib==3.10.0
    >>requirements-cpu.txt echo ultralytics
)

if not exist "requirements-gpu.txt" (
    >requirements-gpu.txt echo numpy==2.2.6
    >>requirements-gpu.txt echo opencv-python==4.10.0.84
    >>requirements-gpu.txt echo Pillow==11.1.0
    >>requirements-gpu.txt echo PySide6==6.8.2
    >>requirements-gpu.txt echo matplotlib==3.10.0
    >>requirements-gpu.txt echo ultralytics
)

if not exist "app\main.py" (
    >app\main.py echo """WallMorph application entry point."""
    >>app\main.py echo.
    >>app\main.py echo import sys
    >>app\main.py echo from PySide6.QtWidgets import QApplication, QLabel
    >>app\main.py echo.
    >>app\main.py echo def main^(^) -^> int:
    >>app\main.py echo     app = QApplication^(sys.argv^)
    >>app\main.py echo     window = QLabel^("WallMorph - Interactive Wall Redesigning System"^)
    >>app\main.py echo     window.setWindowTitle^("WallMorph"^)
    >>app\main.py echo     window.resize^(1000, 650^)
    >>app\main.py echo     window.show^(^)
    >>app\main.py echo     return app.exec^(^)
    >>app\main.py echo.
    >>app\main.py echo if __name__ == "__main__":
    >>app\main.py echo     raise SystemExit^(main^(^)^)
)

echo.
echo [3/7] Creating NVIDIA GPU virtual environment...
if not exist "venv_gpu\Scripts\python.exe" (
    python -m venv venv_gpu
    if errorlevel 1 goto :fail
)

call "venv_gpu\Scripts\activate.bat"

echo.
echo [4/7] Upgrading pip/build tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo.
echo [5/7] Installing PyTorch 2.7.0 + CUDA 12.6...
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :fail

echo.
echo [6/7] Installing WallMorph dependencies...
python -m pip install -r requirements-gpu.txt
if errorlevel 1 goto :fail

echo.
echo [7/7] Verifying installation and CUDA...
python -c "import torch,cv2,numpy,PIL,PySide6,ultralytics; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('PyTorch CUDA:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU DETECTED'); print('OpenCV:',cv2.__version__); print('NumPy:',numpy.__version__); print('Ultralytics:',ultralytics.__version__)"
if errorlevel 1 goto :fail

echo.
echo ==========================================================
echo          NVIDIA GPU SETUP COMPLETED SUCCESSFULLY
echo ==========================================================
echo.
echo Environment: venv_gpu
echo.
echo Activate later:
echo     venv_gpu\Scripts\activate
echo.
echo Run:
echo     python app\main.py
echo.
echo GitHub remote setup is intentionally NOT automatic.
echo.
pause
exit /b 0

:fail
echo.
echo ==========================================================
echo                  SETUP FAILED
echo ==========================================================
echo Review the error above.
echo.
pause
exit /b 1
