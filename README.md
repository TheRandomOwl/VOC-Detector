## Build voc-gui and voc-cli
Install python https://www.python.org/

Use the package manager pip to install the necessary dependencies.

```bash
pip install -r requirements.txt
```

```bash
pyinstaller -F --hidden-import=PIL._tkinter_finder --hide-console minimize-early voc-gui.py
```

```bash
pyinstaller -F --hidden-import=PIL._tkinter_finder voc-cli.py
```

## How to use
to do
