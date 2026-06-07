import sys
from pathlib import Path

IMPORTANT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = IMPORTANT_DIR / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from test_crop_rotate_debug import main


if __name__ == "__main__":
    main()
