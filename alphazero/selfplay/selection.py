"""
selection.py — Selezione PUCT per MCTS guidato da rete.

Formula (validata da ChatGPT, Settimana 4):
    score(child) = Q(child) + c_puct * P(child) * sqrt(N_parent) / (1 + N_child)

Dove:
- Q(child) = W(child) / N(child)  (valore medio dal POV del giocatore del child)
- P(child) = prior dato dalla rete neurale al momento dell'espansione
- N_parent = visite al padre
- N_child  = visite al figlio (0 = mai visitato → forza esplorazione)

Note:
- Per N_child=0, U = c * P * sqrt(N_parent), Q=0 → il PUCT score e' tutto
  esplorazione guidata dalla rete. Standard AlphaZero.
- c_puct=1.5 e' il valore consigliato per partire (ChatGPT).
"""

from __future__ import annotations
from typing import Optional
import math

from .node import Node


def select_child(node: Node, c_puct: float = 1.5) -> Optional[Node]:
    """
    Seleziona il figlio con PUCT score massimo.
    
    Args:
        node: nodo padre (deve avere children non vuoti)
        c_puct: coefficiente di esplorazione (1.5 default, validato ChatGPT)
    
    Returns:
        il figlio scelto, o None se node non ha figli
    
    Convenzione:
    - Q dal POV del giocatore del CHILD (non del parent!)
    - Quando il backup e' fatto correttamente con cambio segno, questo
      e' equivalente al "MIN" del minimax in giochi a 2 giocatori.
    """
    if not node.children:
        return None

    best_score = -float("inf")
    best_child: Optional[Node] = None

    sqrt_parent_N = math.sqrt(node.N) if node.N > 0 else 1.0

    for action, child in node.children.items():
        # Q del child (= valore medio dal POV del giocatore del child)
        Q = child.W / child.N if child.N > 0 else 0.0

        # Esplorazione guidata dal prior della rete
        U = c_puct * child.P * sqrt_parent_N / (1.0 + child.N)

        score = Q + U

        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def select_action_from_root(
    root: Node,
    temperature: float = 1.0,
    rng=None,
) -> tuple[int, list]:
    """
    Sceglie l'azione finale dalla root MCTS, dopo aver fatto le simulazioni.
    
    Convenzione AlphaZero:
    - temperature=1.0  -> sample con probabilita' = visite^1 / sum
                          (esplorazione, primi 30 step della partita)
    - temperature=0    -> argmax delle visite (deterministico)
    
    Args:
        root: root MCTS dopo n_simulations simulazioni
        temperature: 1.0 o 0
        rng: numpy.random.Generator (per riproducibilita')
    
    Returns:
        (azione_scelta, distribuzione_visite_normalizzata)
        La distribuzione viene salvata come policy_target nel buffer.
        
    NB: ChatGPT ha chiarito - SI SALVA LA DISTRIBUZIONE USATA, non si
    riconverte a T=1 dopo. Quindi:
    - se T=1, salviamo la distribuzione "morbida"
    - se T=0, salviamo una distribuzione one-hot
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()

    if not root.children:
        raise ValueError("Root non ha figli — MCTS non e' stato eseguito?")

    actions = list(root.children.keys())
    visits = np.array(
        [root.children[a].N for a in actions],
        dtype=np.float64,
    )

    if temperature == 0 or temperature < 1e-6:
        # Argmax deterministico
        best_idx = int(np.argmax(visits))
        action = actions[best_idx]
        # Distribuzione one-hot per il policy_target
        dist = np.zeros(len(actions))
        dist[best_idx] = 1.0
    else:
        # Sample con temperatura
        visits_t = visits ** (1.0 / temperature)
        if visits_t.sum() == 0:
            # Edge case: tutti i figli con 0 visite → uniforme
            dist = np.ones(len(actions)) / len(actions)
        else:
            dist = visits_t / visits_t.sum()
        action = int(rng.choice(actions, p=dist))

    return action, list(zip(actions, dist))
