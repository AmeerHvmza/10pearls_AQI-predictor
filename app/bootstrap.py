"""Puts the repo root, `src/`, and `app/` on `sys.path`.

Every dashboard module imports this first so `config`, `predict`, and the
sibling page modules resolve the same way whether the app is launched with
`streamlit run`, imported by AppTest, or run from another working directory.
"""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")

for _path in (ROOT_DIR, SRC_DIR, APP_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
