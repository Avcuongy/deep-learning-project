from pathlib import Path
import runpy
import logging

import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
TEST_DIR = PROJECT_ROOT / "test"


def _ensure_directories() -> None:
    for path in (DATA, LOGS_DIR, TEST_DIR):
        path.mkdir(parents=True, exist_ok=True)
        logging.info("Configuration: Initializing folder at %s", path)

def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        filemode="a",
        filename=LOGS_DIR / "config.log",
    )
    logging.info("Configuration: Starting")
    _ensure_directories()
    logging.info("Configuration: Complete")


if __name__ == "__main__":
    main()
