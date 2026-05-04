"""
trainer.py — Training loop minimale per AlphaZero.

Settimana 5, sub-step 2.

Caratteristiche:
- Adam optimizer con weight_decay=1e-4 (L2 regularization)
- Loss AlphaZero standard (MSE + CrossEntropy)
- Save/load checkpoint
- Codice agnostico CPU/GPU (parametro device)

Decisioni di design:
- lr=0.001 di partenza (ChatGPT). Scendere a 0.0003 se instabile.
- gradient clipping a max_norm=1.0 per evitare esplosioni
- step counter interno per logging/scheduling
"""

from __future__ import annotations
from typing import Optional, List
from pathlib import Path
import torch
import torch.optim as optim

from ..network import RisikoNet, alphazero_loss
from ..selfplay.self_play import TrainingSample
from .replay_buffer import samples_to_batch


class Trainer:
    """
    Wrapper minimale per il training di RisikoNet.
    
    Uso tipico:
        trainer = Trainer(net, lr=0.001)
        for step in range(N):
            samples = buffer.sample(batch_size=512)
            loss_dict = trainer.train_step(samples)
            print(loss_dict)
    """

    def __init__(
        self,
        net: RisikoNet,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        grad_clip: Optional[float] = 1.0,
        value_weight: float = 1.0,
        policy_weight: float = 1.0,
    ):
        self.net = net.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.step_counter = 0

    def train_step(self, samples: List[TrainingSample]) -> dict:
        """
        Esegue UN passo di training su un batch di samples.
        
        Args:
            samples: lista di TrainingSample (es. da ReplayBuffer.sample())
        
        Returns:
            dict con metriche del passo:
                - total_loss: float
                - value_loss: float
                - policy_loss: float
                - grad_norm: float (norma del gradiente prima del clip)
        """
        self.net.train()
        
        # Tensorizza il batch
        batch = samples_to_batch(samples, device=self.device)
        
        # Forward
        policy_logits, value_pred = self.net(batch["obs"])
        
        # Loss
        loss_dict = alphazero_loss(
            policy_logits, value_pred,
            batch["policy_target"], batch["value_target"],
            batch["mask"],
            value_weight=self.value_weight,
            policy_weight=self.policy_weight,
        )
        
        # Backward
        self.optimizer.zero_grad()
        loss_dict["total"].backward()
        
        # Gradient clipping (evita esplosioni)
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.net.parameters(),
                max_norm=self.grad_clip,
            )
        else:
            # Calcolo manuale per logging
            grad_norm = torch.tensor(0.0)
            for p in self.net.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2) ** 2
            grad_norm = grad_norm ** 0.5
        
        self.optimizer.step()
        self.step_counter += 1
        
        return {
            "total_loss": float(loss_dict["total"].item()),
            "value_loss": float(loss_dict["value_loss"].item()),
            "policy_loss": float(loss_dict["policy_loss"].item()),
            "grad_norm": float(grad_norm.item()),
            "step": self.step_counter,
        }

    # ─────────────────────────────────────────────────────────────
    #  Checkpoint
    # ─────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> None:
        """Salva stato completo (rete + optimizer + step)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step_counter": self.step_counter,
        }, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        """Carica stato salvato."""
        ckpt = torch.load(str(path), map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step_counter = ckpt.get("step_counter", 0)
