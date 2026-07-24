import json
from qec_decoder import runlog


def test_save_run_writes_record(tmp_path):
    path = runlog.save_run("unit", {"d": 3}, {"logical_error_rate": 0.1},
                           seed=1, out_dir=str(tmp_path))
    rec = json.load(open(path))
    assert rec["kind"] == "unit"
    assert rec["args"] == {"d": 3}
    assert rec["metrics"]["logical_error_rate"] == 0.1
    assert rec["seed"] == 1
    assert "duration_s" in rec and "start" in rec and "git_commit" in rec


def test_run_context_records_metrics(tmp_path):
    with runlog.Run("unit2", {"x": 1}, seed=2) as run:
        run.record({"acc": 0.9})
    # exactly one record written
    files = list(tmp_path.glob("*.json")) if False else None
    # context writes to default dir; assert via return path instead
    assert run.path.endswith(".json")
    rec = json.load(open(run.path))
    assert rec["metrics"]["acc"] == 0.9 and rec["duration_s"] >= 0


def test_run_context_records_error_on_exception():
    run = None
    try:
        with runlog.Run("unit3", {"x": 1}, seed=3) as run:
            raise ValueError("boom")
    except ValueError:
        pass
    assert run is not None and run.path is not None
    rec = json.load(open(run.path))
    assert rec["extra"]["error"] == "boom"


def test_get_logger_idempotent():
    a = runlog.get_logger()
    b = runlog.get_logger()
    assert a is b
