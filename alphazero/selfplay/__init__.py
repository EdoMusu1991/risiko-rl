"""alphazero.selfplay — MCTS guidato da rete neurale per Risiko."""
from .node import Node
from .selection import select_child, select_action_from_root
from .backup import backup
from .simulate import simulate, simulate_simmetrico
from .search import search, search_simmetrico, visite_to_policy_full
from .self_play import gioca_partita_selfplay, gioca_partita_selfplay_simmetrica, TrainingSample
from .parallel import (
    gioca_n_partite_parallele,
    gioca_n_partite_vs_random_parallele,
    gioca_n_partite_match_parallele,
)

__all__ = [
    "Node",
    "select_child",
    "select_action_from_root",
    "backup",
    "simulate",
    "simulate_simmetrico",
    "search",
    "search_simmetrico",
    "visite_to_policy_full",
    "gioca_partita_selfplay",
    "gioca_partita_selfplay_simmetrica",
    "gioca_n_partite_parallele",
    "gioca_n_partite_vs_random_parallele",
    "gioca_n_partite_match_parallele",
    "TrainingSample",
]