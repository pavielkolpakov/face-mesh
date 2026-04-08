import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir():
    sys.path.insert(0, str(_src))

from face_mesh.interfaces.cli import main

if __name__ == "__main__":
    main()
