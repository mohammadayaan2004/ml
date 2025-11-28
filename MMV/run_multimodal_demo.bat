@echo off
echo Running Multimodal Vision-Language Pipeline Demo...
echo.

cd /d "%~dp0"

python MMV/demo.py

echo.
echo Demo completed.
pause