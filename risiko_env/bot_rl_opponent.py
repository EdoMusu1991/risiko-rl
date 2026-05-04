"""
bot_rl_opponent.py — Adapter per usare un modello RL come avversario in self-play.

Permette di far giocare un turno completo a un MaskablePPO caricato, per
un giocatore diverso dal bot principale dell'env.

Strategia: crea un mini-env temporaneo con bot_color=colore_avversario,
innesta lo stesso oggetto stato (per modificarlo in-place), e fa girare
il modello finche' il turno dell'avversario non e' completato.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import random as _random
import sys

import numpy as np

if TYPE_CHECKING:
    from .stato import StatoPartita

MAX_STEP_TURNO_AVVERSARIO = 200


def gioca_turno_rl(stato: "StatoPartita", colore: str, modello, rng: _random.Random) -> None:
    """
    Fa giocare un turno completo al modello RL `modello` per il giocatore `colore`.
    Modifica `stato` in-place esattamente come `gioca_turno_random`.

    Su errore: fallback silente al bot random (per non crashare il training).
    """
    from . import encoding as _encoding

    flag_originale = _encoding.STAGE_A_ATTIVO
    try:
        # Setta flag in base al modello
        dim_modello = modello.observation_space.shape[0]
        if dim_modello == 318:
            _encoding.STAGE_A_ATTIVO = False
        elif dim_modello in (330, 342):
            _encoding.STAGE_A_ATTIVO = True

        _gioca_turno_rl_inner(stato, colore, modello, rng)

    except Exception as e:
        print(
            f"[bot_rl_opponent] errore turno {colore}: {type(e).__name__}: {e}. "
            f"Fallback random.",
            file=sys.stderr,
        )
        from .bot_random import gioca_turno_random
        try:
            gioca_turno_random(stato, colore, rng)
        except Exception as e2:
            print(f"[bot_rl_opponent] anche fallback random fallito: {e2}", file=sys.stderr)
    finally:
        _encoding.STAGE_A_ATTIVO = flag_originale


def _gioca_turno_rl_inner(stato: "StatoPartita", colore: str, modello, rng: _random.Random) -> None:
    from .env import RisikoEnv

    env_temp = RisikoEnv(
        bot_color=colore,
        max_steps=MAX_STEP_TURNO_AVVERSARIO * 2,
        seed=None,
        log_eventi=False,
        avversari=None,
    )

    # Innesto stato e rng condivisi
    env_temp.stato = stato
    env_temp.rng = rng
    env_temp.step_count = 0

    # Self-play: il mini-env NON deve giocare i turni avversari (lo fa il padre).
    # Inoltre _fine_turno_bot del mini-env chiamera' gestisci_fine_turno (pesca + sdadata)
    # — anche questo lo fa il padre, quindi dobbiamo evitarlo. Soluzione: facciamo
    # eseguire al mini-env solo le 4 sotto-fasi (tris/rinforzo/attacco/spostamento),
    # NON il _fine_turno_bot().
    env_temp._skip_giro_avversari = True
    env_temp._skip_fine_turno_bot = True  # nuovo flag

    # Avvia sotto-fase TRIS
    env_temp._inizia_fase_tris()

    # Costruisci observation e info iniziali
    obs = env_temp._costruisci_observation()
    info = env_temp._costruisci_info()

    n_step = 0
    while n_step < MAX_STEP_TURNO_AVVERSARIO:
        if stato.terminata:
            return

        # Se sotto_fase e' tornata None, il turno bot e' finito (le 4 sotto-fasi
        # sono completate). Esci.
        if env_temp.sotto_fase is None:
            return

        mask = info["action_mask"]
        action, _ = modello.predict(obs, action_masks=mask, deterministic=True)
        obs, _reward, term, trunc, info = env_temp.step(int(action))
        n_step += 1

        if stato.terminata or term or trunc:
            return

    print(
        f"[bot_rl_opponent] {colore} ha fatto {n_step} step senza finire turno. Forzo exit.",
        file=sys.stderr,
    )
