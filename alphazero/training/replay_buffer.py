"""
replay_buffer.py — Replay buffer per AlphaZero.

Settimana 5, sub-step 1.

Caratteristiche:
- Finestra scorrevole (deque con maxlen)
- Sample uniforme con batch_size
- Conversione automatica TrainingSample -> tensor batch

Standard AlphaZero:
- Buffer 100k-500k step. 100k = ~200-400 partite Risiko (lunghe).
- Sampling con replacement (uniforme)
- Niente weighted sampling (non serve nelle prime versioni)
"""

from __future__ import annotations
from collections import deque
from typing import List, Optional
import random
import numpy as np
import torch

from ..selfplay.self_play import TrainingSample


class ReplayBuffer:
    """
    Buffer FIFO di TrainingSample. Quando pieno, scarta i piu' vecchi.
    
    Uso:
        buffer = ReplayBuffer(max_size=100_000)
        for partita in tante_partite:
            buffer.add_partita(samples_della_partita)
        batch = buffer.sample(batch_size=512)
    """

    def __init__(self, max_size: int = 100_000, seed: Optional[int] = None):
        self.max_size = max_size
        self.buffer: deque[TrainingSample] = deque(maxlen=max_size)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_partita(self, samples: List[TrainingSample]) -> None:
        """Aggiunge tutti i sample di una partita al buffer."""
        for s in samples:
            self.buffer.append(s)

    def add(self, sample: TrainingSample) -> None:
        """Aggiunge un singolo sample."""
        self.buffer.append(sample)

    def sample(self, batch_size: int) -> List[TrainingSample]:
        """
        Sampling uniforme con replacement. Restituisce una lista di
        TrainingSample (non ancora tensorizzati).
        
        Se batch_size > len(buffer), ritorna comunque batch_size sample
        (con replacement).
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer vuoto, impossibile sample")
        n = min(len(self.buffer), batch_size)
        # Per AlphaZero standard usa with-replacement
        return self.rng.choices(list(self.buffer), k=batch_size)

    def is_ready(self, min_size: int = 1024) -> bool:
        """True se il buffer ha abbastanza sample per iniziare a trainare."""
        return len(self.buffer) >= min_size


# ─────────────────────────────────────────────────────────────────
#  Conversione batch -> tensori PyTorch
# ─────────────────────────────────────────────────────────────────

def samples_to_batch(
    samples: List[TrainingSample],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Converte una lista di TrainingSample in un dict di tensori per il training.
    
    Returns:
        {
            "obs":           (B, 342) float32
            "mask":          (B, 1765) bool
            "policy_target": (B, 1765) float32
            "value_target":  (B, 1)    float32
        }
    """
    batch_size = len(samples)
    
    obs = np.stack([s.obs for s in samples], axis=0)
    mask = np.stack([s.mask for s in samples], axis=0)
    policy_target = np.stack([s.policy_target for s in samples], axis=0)
    value_target = np.array([s.value_target for s in samples], dtype=np.float32)
    
    return {
        "obs": torch.from_numpy(obs).float().to(device),
        "mask": torch.from_numpy(mask).bool().to(device),
        "policy_target": torch.from_numpy(policy_target).float().to(device),
        "value_target": torch.from_numpy(value_target).float().unsqueeze(1).to(device),
    }
