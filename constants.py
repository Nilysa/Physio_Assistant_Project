"""
Shared, side-effect-free constants and the single logging configuration
for the whole application.

Kept in its own module (rather than, say, worker.py) so that both
worker.py and app.py can import from here without either of them
depending on the other -- this is what keeps the import graph acyclic:

    config.py  (no internal deps)
    engine.py  -> config.py
    worker.py  -> constants.py, engine.py
    app.py     -> constants.py, config.py, engine.py, worker.py
    main.py    -> app.py
"""
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
)
logger = logging.getLogger("physio_assistant")

COLOR_GOOD_RGB = (46, 204, 113)  # green
COLOR_BAD_RGB = (231, 76, 60)  # red
if getattr(sys, "frozen", False):
    # Running as a bundled .exe -- anchor to the executable's own folder
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running as a normal .py script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
CSV_FLUSH_EVERY_N_FRAMES = 15
