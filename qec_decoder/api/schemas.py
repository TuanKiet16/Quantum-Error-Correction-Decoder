from pydantic import BaseModel


class InjectReq(BaseModel):
    d: int
    mode: str = "random"           # "manual" | "random"
    errors: list | None = None     # [[qubit, "X"|"Z"], ...]
    p: float | None = None


class DecodeReq(BaseModel):
    d: int
    detection_events: list
    decoder: str = "mwpm"          # "qcnn" | "cnn" | "mwpm"


class BatchReq(BaseModel):
    d: int
    p: float
    shots: int = 2000
    decoder: str = "mwpm"
