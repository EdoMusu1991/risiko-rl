"""alphazero.selfplay — MCTS guidato da rete neurale per Risiko."""
from .node import Node
from .selection import select_child, select_action_from_root
from .backup import backup
from .simulate import simulate
from .search import search, visite_to_policy_full
from .self_play import gioca_partita_selfplay, TrainingSample

__all__ = [
    "Node",
    "select_child",
    "select_action_from_root",
    "backup",
    "simulate",
    "search",
    "visite_to_policy_full",
    "gioca_partita_selfplay",
    "TrainingSample",
]
