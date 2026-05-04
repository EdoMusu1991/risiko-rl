"""
backup.py — Backup MCTS con cambio segno per giocatori multipli.

Validato da ChatGPT (Settimana 4):
- Confronto con player_LEAF (fisso), NON con last_player che cambia
- Perche': in Risiko sequenze tipo BLU-BLU-BLU-ROSSO-ROSSO non sono
  alternate, quindi "inverti quando cambia giocatore" e' fragile
- Versione corretta: ogni nodo confronta il SUO player_to_move con quello
  della FOGLIA (player_leaf), e segna v o -v di conseguenza

Ricorda: il "value" della rete e' dal POV del giocatore della FOGLIA,
quindi e' positivo per quel giocatore. Quando risali nell'albero:
- Nodo dello STESSO giocatore della foglia → value ha lo stesso segno
- Nodo di un AVVERSARIO della foglia → value ha segno opposto
  (perche' "buono per la foglia" = "cattivo per l'avversario")
"""

from __future__ import annotations
from typing import List

from .node import Node


def backup(path: List[Node], value: float, player_leaf: str) -> None:
    """
    Propaga il valore lungo il path della simulazione MCTS.
    
    Args:
        path: lista di nodi dalla ROOT alla FOGLIA (inclusi entrambi)
        value: valore stimato dalla rete (o terminale) dal POV del player_leaf
        player_leaf: chi muove nel nodo foglia (es. "BLU" o "ROSSO")
    
    Convenzione AlphaZero:
    - W di ogni nodo = somma di valori dal POV del player_to_move di QUEL nodo
    - Quindi:
        * Se node.player_to_move == player_leaf: W += value
        * Altrimenti:                            W += -value
    
    NB: questa funzione e' deterministica e non chiama mai env, rete, o
    altre funzioni esterne. Modifica solo i nodi del path.
    """
    for node in reversed(path):
        if node.player_to_move == player_leaf:
            v = value
        else:
            v = -value
        node.N += 1
        node.W += v
