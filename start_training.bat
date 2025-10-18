@echo off
echo ============================================
echo Solar Panel Dust Detection Training
echo ============================================
echo.

echo Activating virtual environment...
call solar_env\Scripts\activate.bat

echo.
echo Starting training with your RTX 3050...
python train_solar_dust_detector.py

echo.
echo Training completed! Check the models folder for results.
pause