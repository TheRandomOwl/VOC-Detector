#!/bin/bash

# install dependencies
pip install -r requirements.txt

# build programs
pyinstaller -F --hidden-import=PIL._tkinter_finder voc-gui.py
pyinstaller -F --hidden-import=PIL._tkinter_finder voc-cli.py