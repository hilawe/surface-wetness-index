"""pytest bootstrap: make the swi package importable and ensure libswi is built."""

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))


def _ensure_lib():
    for name in ("libswi.dylib", "libswi.so"):
        if os.path.exists(os.path.join(_SRC, name)):
            return
    subprocess.run(["make"], cwd=_SRC, check=True)


_ensure_lib()
