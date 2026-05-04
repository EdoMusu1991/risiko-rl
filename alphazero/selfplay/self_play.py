"""
self_play.py — Gioca una partita completa con MCTS+rete e raccoglie dati.

Sub-step 5b della Settimana 4.

NOTA SETTIMANA 6: questa versione e' INCOMPLETA per AlphaZero puro.
L'env oggi salta automaticamente i turni avversari giocandoli con bot interno
(random/euristico). Quindi self-play raccoglie sample SOLO dal bot master (BLU).
Il fix per simmetria vera e' in lavorazione.
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


TEMPERATURE_DROP_STEP = 30


@dataclass
class TrainingSample:
    obs: np.ndarray
    mask: np.ndarray
    policy_target: np.ndarray
    player_at_state: str
    value_target: float = 0.0


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
    if rng is None:
        rng = np.random.default_rng(seed)
    
    obs, info = env.reset(seed=seed)
    samples: List[TrainingSample] = []
    n_decisioni = 0
    
    while True:
        if n_decisioni >= max_decisioni:
            break
        
        mask = info["action_mask"]
        n_legali = int(mask.sum())
        
        if n_legali == 0:
            break
        if n_legali == 1:
            azione = int(np.where(mask)[0][0])
            obs, reward, term, trunc, info = env.step(azione)
            if term or trunc:
                break
            continue
        
        player = env.stato.giocatore_corrente
        root = Node(
            snapshot=env.snapshot(),
            player_to_move=player,
            P=1.0,
        )
        
        T = 1.0 if n_decisioni < temperature_drop_step else 0.0
        
        action, _ = search(
            root, env, net,
            n_simulations=n_simulations,
            c_puct=c_puct,
            temperature=T,
            rng=rng,
        )
        
        policy_target = visite_to_policy_full(root, ACTION_DIM, temperature=T)
        
        samples.append(TrainingSample(
            obs=obs.copy(),
            mask=mask.copy(),
            policy_target=policy_target,
            player_at_state=player,
            value_target=0.0,
        ))
        
        obs, reward, term, trunc, info = env.step(action)
        n_decisioni += 1
        
        if verbose and n_decisioni % 50 == 0:
            print(f"    decisione {n_decisioni}, sotto_fase={env.sotto_fase}, "
                  f"player={env.stato.giocatore_corrente}")
        
        if term or trunc:
            break
    
    reward_finale = float(reward)
    bot_color = env.bot_color
    
    for sample in samples:
        if sample.player_at_state == bot_color:
            sample.value_target = reward_finale
        else:
            sample.value_target = -reward_finale
    
    stats = {
        "n_decisioni_mcts": n_decisioni,
        "vincitore": info.get("vincitore"),
        "motivo_fine": info.get("motivo_fine"),
        "reward_finale": reward_finale,
        "n_samples": len(samples),
    }
    
    return samples, stats
