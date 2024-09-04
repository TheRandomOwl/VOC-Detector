#!/bin/bash

# install dependencies
pip install -r requirements.txt

# build programs
pyinstaller -F --clean --noconfirm --hidden-import=tkinterk voc-gui.py
pyinstaller -F --clean --noconfirm --hidden-import=PIL._tkinter_finder voc-cli.py