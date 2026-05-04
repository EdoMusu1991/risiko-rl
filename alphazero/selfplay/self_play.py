"""
self_play.py — Gioca una partita completa con MCTS+rete e raccoglie dati.

Sub-step 5b della Settimana 4.

Una partita di self-play produce una lista di "TrainingSample":
- stato (obs)
- mask (action mask)
- policy_target (distribuzione visite MCTS)
- player_at_state (chi muoveva — serve per assegnare value_target)
- value_target (riempito a fine partita: reward dal POV di player_at_state)

In 1v1 simmetrico:
- Stessa rete per entrambi i giocatori
- Due alberi MCTS separati (uno per BLU, uno per ROSSO)
- Tutti gli stati di entrambi i giocatori finiscono nel dataset

Schedule temperature (standard AlphaZero):
- T=1 nei primi N step della partita (esplorazione)
- T=0 dopo (sfruttamento)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch

from risiko_env import RisikoEnv
from .node import Node
from .search import search, visite_to_policy_full
from ..network import RisikoNet, ACTION_DIM


# Step entro cui T=1 (esplorazione). Standard AlphaZero scacchi: 30.
# Risiko ha turni piu' lunghi, ma 30 step sono ~1-2 turni del bot. Ok per partire.
TEMPERATURE_DROP_STEP = 30


@dataclass
class TrainingSample:
    """Un campione del dataset di self-play (una decisione MCTS)."""
    obs: np.ndarray              # (342,) observation
    mask: np.ndarray             # (1765,) action mask bool
    policy_target: np.ndarray    # (1765,) distribuzione visite normalizzata
    player_at_state: str         # "BLU" o "ROSSO" (per assegnare value_target)
    value_target: float = 0.0    # riempito a fine partita


def gioca_partita_selfplay(
    env: RisikoEnv,
    net: RisikoNet,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    temperature_drop_step: int = TEMPERATURE_DROP_STEP,
    seed: Optional[int] = None,
    max_decisioni: int = 2000,
    verbose: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[List[TrainingSample], dict]:
    """
    Gioca una partita 1v1 completa con self-play e raccoglie dati per il
    replay buffer.
    
    Args:
        env: RisikoEnv (verra' resettata)
        net: RisikoNet condivisa fra i due "giocatori"
        n_simulations: simulazioni MCTS per decisione (50 default)
        c_puct: coefficiente PUCT
        temperature_drop_step: step entro cui T=1, dopo T=0
        seed: per env.reset()
        max_decisioni: safety cap (alcune partite Risiko sono lunghe)
        verbose: log progress
        rng: numpy generator
    
    Returns:
        samples: lista di TrainingSample con value_target gia' riempito
        stats: dict con info diagnostiche (n_step, vincitore, durata, ...)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    
    obs, info = env.reset(seed=seed)
    
    samples: List[TrainingSample] = []
    n_decisioni = 0
    
    while True:
        if n_decisioni >= max_decisioni:
            if verbose:
                print(f"  Max decisioni raggiunto ({max_decisioni})")
            break
        
        mask = info["action_mask"]
        n_legali = int(mask.sum())
        
        # FAST-PATH: se solo 1 azione legale, salta MCTS (deterministica)
        # Non e' una "vera decisione" del bot, non la salviamo nel buffer
        if n_legali == 0:
            break
        if n_legali == 1:
            azione = int(np.where(mask)[0][0])
            obs, reward, term, trunc, info = env.step(azione)
            if term or trunc:
                break
            continue
        
        # MCTS+rete su decisione vera
        player = env.stato.giocatore_corrente
        root = Node(
            snapshot=env.snapshot(),
            player_to_move=player,
            P=1.0,
        )
        
        # Temperature schedule
        T = 1.0 if n_decisioni < temperature_drop_step else 0.0
        
        # Esegui search
        action, _ = search(
            root, env, net,
            n_simulations=n_simulations,
            c_puct=c_puct,
            temperature=T,
            rng=rng,
        )
        
        # Estrai policy_target su tutto l'action space (vettore 1765-D)
        # NB: ChatGPT - usiamo la distribuzione USATA (con T applicato)
        policy_target = visite_to_policy_full(root, ACTION_DIM, temperature=T)
        
        # Salva il sample (value_target riempito a fine partita)
        samples.append(TrainingSample(
            obs=obs.copy(),
            mask=mask.copy(),
            policy_target=policy_target,
            player_at_state=player,
            value_target=0.0,  # placeholder
        ))
        
        # Esegui l'azione scelta
        obs, reward, term, trunc, info = env.step(action)
        n_decisioni += 1
        
        if verbose and n_decisioni % 50 == 0:
            print(f"    decisione {n_decisioni}, sotto_fase={env.sotto_fase}, "
                  f"player={env.stato.giocatore_corrente}")
        
        if term or trunc:
            break
    
    # ────────────────────────────────────────────────────────────────
    # Riempimento value_target a fine partita
    # ────────────────────────────────────────────────────────────────
    # reward finale e' dal POV di env.bot_color (BLU). Convenzione:
    # - se BLU vince/pareggia: reward >= 0 → BLU vede +reward, ROSSO vede -reward
    # - se BLU perde: reward < 0 → BLU vede reward (negativo), ROSSO vede -reward
    reward_finale = float(reward)
    bot_color = env.bot_color  # "BLU" in 1v1 standard
    
    for sample in samples:
        if sample.player_at_state == bot_color:
            sample.value_target = reward_finale
        else:
            sample.value_target = -reward_finale
    
    # Stats
    stats = {
        "n_decisioni_mcts": n_decisioni,
        "vincitore": info.get("vincitore"),
        "motivo_fine": info.get("motivo_fine"),
        "reward_finale": reward_finale,
        "n_samples": len(samples),
    }
    
    return samples, stats
