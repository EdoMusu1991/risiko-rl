"""
search.py — Wrapper di alto livello per MCTS+rete.

Esegue N simulazioni dalla root e produce:
- L'azione scelta (sampleata o argmax in base a temperature)
- La distribuzione delle visite (= policy_target per il replay buffer)

Sub-step 5a della Settimana 4.

NB ChatGPT (Settimana 4):
- Salva la distribuzione USATA, non riconvertire dopo
- Temperature schedule: T=1 primi 30 step, T=0 dopo
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .node import Node
from .simulate import simulate
from .selection import select_action_from_root
from ..network import RisikoNet


def search(
    root: Node,
    env,
    net: RisikoNet,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[int, list]:
    """
    Esegue n_simulations simulazioni MCTS dalla root, poi estrae:
    - l'azione da giocare (sampling con temperature)
    - la distribuzione delle visite (per il replay buffer)
    
    Args:
        root: Node radice. DEVE avere snapshot e player_to_move impostati.
        env: RisikoEnv. Verra' restituito allo stato della root al termine.
        net: RisikoNet usata per policy/value
        n_simulations: numero di simulazioni MCTS (50-200 standard)
        c_puct: coefficiente esplorazione PUCT (1.5)
        temperature: 0 = argmax, 1.0 = sampling proporzionale
        rng: numpy random Generator (per riproducibilita')
    
    Returns:
        (action, dist):
            action: int — azione scelta da giocare nell'env
            dist: list di (action_int, prob_float) — distribuzione visite,
                  da salvare come policy_target nel replay buffer
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Esegui N simulazioni
    for _ in range(n_simulations):
        simulate(root, env, net, c_puct=c_puct)
    
    # Estrai azione + distribuzione visite
    # NB: select_action_from_root salva ESATTAMENTE la distribuzione usata
    # (one-hot se T=0, morbida se T>0). Questo e' il policy_target per training.
    action, dist = select_action_from_root(root, temperature=temperature, rng=rng)
    
    return action, dist


def visite_to_policy_full(
    root: Node,
    action_dim: int = 1765,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Converte le visite della root in una distribuzione di policy_target
    su TUTTO l'action space (1765 dimensioni), con zeri sulle azioni
    illegali. Pronta per essere salvata nel replay buffer.
    
    NB: questa rappresentazione e' ridondante con `dist` di search(),
    ma in formato comodo per il batch training (vettore fixed-size 1765).
    """
    policy = np.zeros(action_dim, dtype=np.float32)
    
    if not root.children:
        return policy
    
    visits = np.array(
        [(a, root.children[a].N) for a in root.children.keys()],
        dtype=np.float64,
    )
    actions, ns = visits[:, 0].astype(int), visits[:, 1]
    
    if temperature == 0 or temperature < 1e-6:
        # One-hot sull'azione piu' visitata
        best = int(actions[np.argmax(ns)])
        policy[best] = 1.0
    else:
        ns_t = ns ** (1.0 / temperature)
        if ns_t.sum() == 0:
            # Edge case: uniformare
            policy[actions] = 1.0 / len(actions)
        else:
            policy[actions] = ns_t / ns_t.sum()
    
    return policy


# ═════════════════════════════════════════════════════════════════════════
#  PR2 — SEARCH SIMMETRICO
# ═════════════════════════════════════════════════════════════════════════

def search_simmetrico(
    root: Node,
    envs: dict,
    net: RisikoNet,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[int, list]:
    """
    Versione simmetrica di search() per AlphaZero puro (PR2).

    Identica a search() ma chiama simulate_simmetrico (che usa due env templati
    e gestisce correttamente i turni avversari come livelli MIN dell'albero).

    Args:
        root: Node radice. DEVE avere snapshot e player_to_move impostati,
            e snapshot deve essere compatibile con envs[root.player_to_move].
        envs: dict {"BLU": env_blu, "ROSSO": env_rosso}, entrambi con
            _skip_giro_avversari=True. Verranno restorati allo stato della
            root al termine.
        net: RisikoNet usata per policy/value.
        n_simulations: numero di simulazioni MCTS.
        c_puct: coefficiente esplorazione PUCT.
        temperature: 0=argmax, 1.0=sampling proporzionale.
        rng: numpy random Generator (per riproducibilita').

    Returns:
        (action, dist) come search().
    """
    from .simulate import simulate_simmetrico

    if rng is None:
        rng = np.random.default_rng()

    for _ in range(n_simulations):
        simulate_simmetrico(root, envs, net, c_puct=c_puct)

    action, dist = select_action_from_root(root, temperature=temperature, rng=rng)
    return action, dist
