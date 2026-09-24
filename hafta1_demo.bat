@echo off
echo ===== Ortam Dogrulama =====
venv\Scripts\python.exe src\check_env.py
echo.
echo Devam etmek icin bir tusa bas...
pause > nul
echo.
echo ===== Baseline YOLOv8 Inference ve FPS Olcumu =====
venv\Scripts\python.exe src\baseline_inference.py
echo.
echo Tamamlandi. Cikmak icin bir tusa bas...
pause > nul
