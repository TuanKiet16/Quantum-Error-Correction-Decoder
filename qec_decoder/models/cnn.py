import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class CNNDecoder(nn.Module):
    """Classical 1D-CNN over the flat detection-event vector."""

    # Classical: one forward per sample, no per-patch quantum-batch blow-up.
    circuits_per_sample = 1

    def __init__(self, n_detectors: int, detector_order=None):
        super().__init__()
        perm = (torch.arange(n_detectors) if detector_order is None
                else torch.as_tensor(detector_order, dtype=torch.long))
        self.register_buffer("det_perm", perm)   # geometry ordering, saved in ckpt
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 4, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.det_perm]     # reorder detectors by lattice geometry
        x = x.unsqueeze(1)          # [B, 1, D]
        x = self.conv(x)
        return self.head(x)
