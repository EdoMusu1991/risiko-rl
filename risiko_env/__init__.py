"""
risiko_env — Simulatore RisiKo per Reinforcement Learning.

Pacchetto che implementa il regolamento RisiKo (variante torneo italiano)
come ambiente Gymnasium per addestramento di bot RL.

Specifica di riferimento: risiko_specifica_v1.2.md
"""

from . import data
from . import stato
from . import setup
from . import combattimento
from . import obiettivi
from . import motore
from . import sdadata
from . import encoding
from . import azioni
from . import env
from . import wrapper

# Esposizione classi principali
from .env import RisikoEnv
from .wrapper import MaskableEnvWrapper

__version__ = "0.6.0-modulo6"
