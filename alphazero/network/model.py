"""
model.py — Rete neurale AlphaZero per Risiko.

Architettura (validata da ChatGPT, Settimana 3 di v2):
- Input: 342 feature normalizzate [0, 1] (Stage A2 attivo)
- Tronco condiviso: Dense(256) -> Dense(256) -> Dense(128) con ReLU
- Policy head: Dense(256) -> Dense(1765) con logits (softmax applicato fuori)
- Value head: Dense(64) -> Dense(1) con tanh -> [-1, +1]
- ~600k parametri totali

Decisioni progettuali:
- MLP semplice (no GNN, no CNN per ora). Se si limita, in Mese 2-3 valutiamo upgrade.
- Xavier init sulle teste finali (Glorot uniform), default He sul tronco
- Action mask applicata FUORI dal modello (in MCTS), non in forward pass

Validazione ChatGPT:
- Architettura approvata
- 256 -> 256 -> 128 (non 256 -> 256 -> 256) per ridurre overfitting
- Loss function MSE+CrossEntropy+L2 standard AlphaZero
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Costanti dimensionali del problema Risiko
INPUT_DIM = 342      # observation space (con Stage A2 attivo)
ACTION_DIM = 1765    # action space totale di Risiko


class RisikoNet(nn.Module):
    """
    Rete neurale a doppia testa: policy + value.
    
    Input: tensor(batch, 342) di feature normalizzate [0, 1]
    Output:
        policy_logits: tensor(batch, 1765) — NON softmaxati
                       (mascherare e softmaxare fuori dal modello)
        value: tensor(batch, 1) — già in [-1, +1] grazie a tanh
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        action_dim: int = ACTION_DIM,
        trunk_sizes: tuple = (256, 256, 128),
        policy_hidden: int = 256,
        value_hidden: int = 64,
    ):
        super().__init__()

        # ─── TRONCO CONDIVISO ─────────────────────────────────────
        # Estrae feature comuni che servono sia a policy che a value.
        # 256 -> 256 -> 128 (modifica di ChatGPT vs il mio 256-256-256
        # iniziale: riduce overfitting nelle prime epoche).
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_sizes[0], trunk_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_sizes[1], trunk_sizes[2]),
            nn.ReLU(inplace=True),
        )

        trunk_out_dim = trunk_sizes[-1]  # 128

        # ─── POLICY HEAD ──────────────────────────────────────────
        # Produce logit per ognuna delle 1765 azioni possibili.
        # NON applichiamo softmax qui: serve fuori, dopo aver
        # mascherato le azioni illegali con -1e9 (NON -inf,
        # raccomandazione di ChatGPT per evitare NaN).
        self.policy_head = nn.Sequential(
            nn.Linear(trunk_out_dim, policy_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(policy_hidden, action_dim),
        )

        # ─── VALUE HEAD ───────────────────────────────────────────
        # Stima il "valore" della posizione corrente per chi muove.
        # Output in [-1, +1] grazie a tanh. Convenzione AlphaZero:
        # +1 = sicuro vittoria, -1 = sicuro sconfitta, 0 = parita'.
        self.value_head = nn.Sequential(
            nn.Linear(trunk_out_dim, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

        # Inizializzazione consigliata da ChatGPT
        self._init_weights()

    def _init_weights(self):
        """
        Inizializzazione robusta:
        - Xavier (Glorot) sulle teste finali per stabilita'
        - Default PyTorch (He per ReLU) sul tronco
        """
        # Trunk: lascia default He (kaiming) che e' adatto a ReLU
        # Final layers: Xavier uniform per output piu' "centrati"
        for module in [self.policy_head[-1], self.value_head[-2]]:
            # policy_head[-1] = ultimo Linear (1765 output)
            # value_head[-2]  = penultimo (Linear prima del Tanh)
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: tensor di shape (batch, INPUT_DIM) o (INPUT_DIM,) per single sample
        
        Returns:
            policy_logits: tensor (batch, ACTION_DIM) - NON softmaxato
            value: tensor (batch, 1) in [-1, +1]
        """
        # Aggiungi batch dim se manca (single-sample inference)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

    def num_parameters(self) -> int:
        """Numero totale di parametri allenabili."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────
#  HELPER: applica action mask
# ─────────────────────────────────────────────────────────────────────

def apply_mask_and_softmax(
    policy_logits: torch.Tensor,
    action_mask: torch.Tensor,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """
    Applica action mask e softmax ai logit della policy.
    
    ATTENZIONE: usiamo -1e9 (NON -inf) per evitare NaN in softmax,
    come raccomandato da ChatGPT.
    
    Args:
        policy_logits: (batch, ACTION_DIM) o (ACTION_DIM,)
        action_mask: stessa shape, bool o float (1.0 = legale, 0.0 = illegale)
        mask_value: valore con cui sostituire i logit illegali
    
    Returns:
        policy: (batch, ACTION_DIM) distribuzione di probabilita'
                (sum=1 sulle legali, 0 sulle illegali)
    """
    if action_mask.dtype != torch.bool:
        action_mask = action_mask.bool()

    masked_logits = policy_logits.masked_fill(~action_mask, mask_value)
    policy = F.softmax(masked_logits, dim=-1)
    return policy


# ─────────────────────────────────────────────────────────────────────
#  HELPER: loss AlphaZero
# ─────────────────────────────────────────────────────────────────────

def alphazero_loss(
    policy_logits_pred: torch.Tensor,
    value_pred: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    action_mask: torch.Tensor,
    value_weight: float = 1.0,
    policy_weight: float = 1.0,
) -> dict:
    """
    Calcola la loss AlphaZero standard.
    
    Loss = value_weight * MSE(v_pred, v_target)
         + policy_weight * CrossEntropy(p_pred, p_target)
    
    L2 regularization gestita dall'optimizer (weight_decay).
    
    Args:
        policy_logits_pred: (batch, ACTION_DIM) — logit grezzi della rete
        value_pred:    (batch, 1) — output value della rete
        policy_target: (batch, ACTION_DIM) — distribuzione MCTS visite
                       (e' gia' una distribuzione, somma 1, zero su illegali)
        value_target:  (batch, 1) — reward finale dal POV del bot in [-1, +1]
        action_mask:   (batch, ACTION_DIM) — azioni legali in quello stato
        value_weight, policy_weight: bilanciamento delle due loss
    
    Returns:
        dict con loss totali e componenti separate per logging
    """
    # Value loss: MSE
    value_loss = F.mse_loss(value_pred, value_target)

    # Policy loss: cross-entropy con mascheramento
    # Applichiamo -1e9 alle azioni illegali, poi log_softmax
    if action_mask.dtype != torch.bool:
        action_mask = action_mask.bool()
    masked_logits = policy_logits_pred.masked_fill(~action_mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    # CrossEntropy: -sum(target * log_pred), media sul batch
    policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()

    total = value_weight * value_loss + policy_weight * policy_loss

    return {
        "total": total,
        "value_loss": value_loss.detach(),
        "policy_loss": policy_loss.detach(),
    }
