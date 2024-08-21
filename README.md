# How to build voc-gui.exe and voc-cli.exe
- `pip install -r requirements.txt`
- `pyinstaller -F --hidden-import=PIL._tkinter_finder --hide-console minimize-early voc-gui.py`
- `pyinstaller -F --hidden-import=PIL._tkinter_finder voc-cli.py`
