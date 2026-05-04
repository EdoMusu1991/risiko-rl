"""alphazero.network — rete neurale per AlphaZero Risiko."""
from .model import (
    RisikoNet,
    INPUT_DIM,
    ACTION_DIM,
    apply_mask_and_softmax,
    alphazero_loss,
)

__all__ = [
    "RisikoNet",
    "INPUT_DIM",
    "ACTION_DIM",
    "apply_mask_and_softmax",
    "alphazero_loss",
]
