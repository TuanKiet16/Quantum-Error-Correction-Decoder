import pytest
from qec_decoder import google_data


def test_ensure_present_raises_when_missing(tmp_path):
    missing = str(tmp_path / "nope.zip")
    with pytest.raises(FileNotFoundError):
        google_data.ensure_present(missing)
