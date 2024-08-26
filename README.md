## Build voc-gui and voc-cli for Windows
Install python https://www.python.org/
Select Add python.exe to PATH

Use the package manager pip to install the necessary dependencies.

```bash
pip install -r requirements.txt

pyinstaller -F --hidden-import=PIL._tkinter_finder --hide-console minimize-early voc-gui.py

pyinstaller -F --hidden-import=PIL._tkinter_finder voc-cli.py
```

## How to use
to do
