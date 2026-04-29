"""
wrapper.py — Wrapper per compatibilità con MaskablePPO.

MaskablePPO di sb3-contrib si aspetta che l'environment esponga un metodo
`action_masks()` che restituisce la maschera delle azioni legali.

Il nostro RisikoEnv mette la maschera in info["action_mask"]. Questo wrapper
la espone anche come metodo della classe, come richiesto da MaskablePPO.

Uso:
    env = RisikoEnv(seed=42)
    env = MaskableEnvWrapper(env)
    # adesso env.action_masks() funziona
"""

import gymnasium as gym
import numpy as np

from .env import RisikoEnv


class MaskableEnvWrapper(gym.Wrapper):
    """
    Wrapper che espone action_masks() come metodo, per MaskablePPO.

    Mantiene tutta l'API standard di Gymnasium, in più aggiunge:
    - action_masks() → np.ndarray bool, maschera delle azioni legali
    """

    def __init__(self, env: RisikoEnv):
        super().__init__(env)
        # tieni un riferimento esplicito per accedere a get_action_mask
        self._risiko_env = env

    def action_masks(self) -> np.ndarray:
        """
        Restituisce la maschera delle azioni legali nello stato corrente.
        Usato da MaskablePPO per filtrare le azioni illegali.
        """
        return self._risiko_env.get_action_mask()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
