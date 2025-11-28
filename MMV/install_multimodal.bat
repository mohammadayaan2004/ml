@echo off
echo Installing Multimodal Vision-Language Pipeline Dependencies...
echo.

cd /d "%~dp0\MMV"

echo Installing Python packages...
pip install -r requirements.txt

echo.
echo Downloading SAM model checkpoint...
echo Note: This may take a few minutes depending on your internet connection.
powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth' -OutFile 'sam_vit_h_4b8939.pth'"

echo.
echo Installation completed!
echo To run the demo, execute run_multimodal_demo.bat
pause