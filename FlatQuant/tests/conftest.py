import sys
import os

# Make ahcptq importable (lives alongside FlatQuant)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent = os.path.dirname(_repo_root)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
