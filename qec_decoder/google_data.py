import os
import numpy as np

GOOGLE_ZIP = "data/google_105Q_surface_code_d3_d5_d7.zip"

_MISSING_MSG = (
    "Google dataset not found at {path}. This 5.7 GB file must be downloaded "
    "manually from https://zenodo.org/records/13273331 "
    "(google_105Q_surface_code_d3_d5_d7.zip). Stopping — will not download it."
)


def ensure_present(path: str = GOOGLE_ZIP) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(_MISSING_MSG.format(path=path))


def load(path: str = GOOGLE_ZIP, d: int = 3):
    ensure_present(path)
    # Parsing is defined by the README INSIDE the zip. When the file is present,
    # read that README first, then map its stim .dem / detection records into
    # (detection_events: bool[N,D], observable_flips: bool[N,L]). Left as a
    # documented stub until the data is available (Part 6 is optional/last).
    raise NotImplementedError(
        "Read the README inside the zip to implement parsing (do not guess)."
    )
