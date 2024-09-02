@echo off

REM Install Python packages from requirements.txt
pip install -r requirements.txt

REM Create standalone executable for voc-gui.py
pyinstaller --hide-console minimize-early voc-gui.py

REM Create standalone executable for voc-cli.py
pyinstaller voc-cli.py

echo Finished creating executables.
pause
