import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION = REPO_ROOT / "alphaapollo" / "core" / "generation"

for p in (REPO_ROOT, GENERATION):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
