@echo off
echo ============================================================
echo    Solar Panel Dust Detection - GPU + Real-Time Camera
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "solar_env" (
    echo ❌ Virtual environment 'solar_env' not found!
    echo Please create it first:
    echo    python -m venv solar_env
    echo    solar_env\Scripts\activate
    echo    pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call solar_env\Scripts\activate

REM Check if we're in the right directory
if not exist "Detect_solar_dust" (
    echo ❌ Dataset directory 'Detect_solar_dust' not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

echo ✅ Environment activated successfully!
echo.

REM Main menu
:menu
echo ============================================================
echo                    MAIN MENU
echo ============================================================
echo 1. Verify GPU Setup
echo 2. Train Model (GPU Accelerated)
echo 3. Run Real-Time Camera Detection
echo 4. Install Dependencies
echo 5. Exit
echo ============================================================
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto gpu_verify
if "%choice%"=="2" goto train_model
if "%choice%"=="3" goto camera_detection
if "%choice%"=="4" goto install_deps
if "%choice%"=="5" goto exit
echo Invalid choice. Please try again.
goto menu

:gpu_verify
echo.
echo 🔍 Verifying GPU setup...
python gpu_verification.py
echo.
pause
goto menu

:train_model
echo.
echo 🚀 Starting GPU-accelerated training...
python gpu_training.py
echo.
pause
goto menu

:camera_detection
echo.
echo 📹 Starting real-time camera detection...
echo Make sure your camera is connected!
python realtime_camera_detection.py
echo.
pause
goto menu

:install_deps
echo.
echo 📦 Installing dependencies...
pip install -r requirements.txt
echo.
echo ✅ Dependencies installed!
pause
goto menu

:exit
echo.
echo 👋 Thank you for using Solar Panel Dust Detection!
echo.
deactivate
exit /b 0
