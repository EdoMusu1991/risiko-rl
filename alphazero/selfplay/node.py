"""
node.py — Node MCTS per AlphaZero Risiko.

Validato da ChatGPT (Settimana 4):
- player_to_move OBBLIGATORIO (Risiko ha sequenze di step stesso giocatore)
- action_taken consigliato (debug + path)
- snapshot salvato sempre (env veloce 1ms)
- statistiche N/W/P standard PUCT

NB: NON questo modulo gestisce backup o selection — solo struttura dati.
La logica e' in mcts.py (selection PUCT) e backup.py.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np


class Node:
    """
    Nodo dell'albero MCTS guidato da rete neurale.
    
    Convenzioni AlphaZero:
    - W e' la SOMMA dei valori, Q = W/N e' la media
    - W e' SEMPRE dal punto di vista del player_to_move di QUESTO nodo
    - P e' il prior dato dalla rete (probabilita' a priori dell'azione che ha
      portato qui — settato dal padre durante l'espansione)
    
    NB: usiamo __slots__ per ridurre memoria (alberi MCTS possono avere
    migliaia di nodi).
    """

    __slots__ = (
        # Statistiche
        "N", "W", "P",
        # Struttura
        "children", "parent", "action_taken",
        # Stato
        "snapshot", "legal_actions", "is_terminal", "terminal_value",
        # Critico per backup con segno
        "player_to_move",
    )

    def __init__(
        self,
        P: float = 0.0,
        parent: Optional["Node"] = None,
        action_taken: Optional[int] = None,
        player_to_move: Optional[str] = None,
        snapshot=None,
    ):
        # Statistiche MCTS
        self.N: int = 0
        self.W: float = 0.0
        self.P: float = P  # prior dato dal padre durante l'espansione
        
        # Struttura albero
        self.children: dict[int, "Node"] = {}
        self.parent: Optional["Node"] = parent
        self.action_taken: Optional[int] = action_taken
        
        # Stato del gioco in questo nodo
        self.snapshot = snapshot  # impostato durante l'espansione (o subito per la root)
        self.legal_actions: Optional[List[int]] = None
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0
        
        # CRITICO: chi muove in questo nodo (per backup con cambio segno)
        self.player_to_move: Optional[str] = player_to_move

    # ─────────────────────────────────────────────────────────────────
    #  Helper read-only
    # ─────────────────────────────────────────────────────────────────

    @property
    def Q(self) -> float:
        """
        Valore medio del nodo. Convenzione: W e' sempre dal POV di
        player_to_move di QUESTO nodo, quindi Q lo e' anche.
        
        Per nodi mai visitati (N=0) ritorna 0 (= valutazione neutra).
        """
        return self.W / self.N if self.N > 0 else 0.0

    def is_expanded(self) -> bool:
        """Vero se il nodo ha figli (e' stato espanso)."""
        return len(self.children) > 0

    def is_leaf(self) -> bool:
        """Vero se il nodo non ha figli (foglia da espandere o terminale)."""
        return len(self.children) == 0

    def __repr__(self) -> str:
        Q_str = f"{self.Q:+.3f}" if self.N > 0 else "n/a"
        return (
            f"Node(player={self.player_to_move}, N={self.N}, "
            f"Q={Q_str}, P={self.P:.3f}, "
            f"action_taken={self.action_taken}, "
            f"children={len(self.children)})"
        )
