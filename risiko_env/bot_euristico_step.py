"""
bot_euristico_step.py - Bot euristico step-by-step per behavior cloning.

Versione step-by-step di bot_euristico.py: invece di giocare un turno intero
modificando lo stato direttamente, riceve un env (in qualunque sotto-fase) e
ritorna un singolo indice di azione per quella sotto-fase.

Riusa la stessa logica delle 3 regole atomiche di bot_euristico:
  1. Attacca solo se win_prob >= SOGLIA_WIN_PROB (0.55)
  2. 60% preferenza per rinforzo confini, 40% uniforme
  3. Lascia almeno MIN_ARMATE_DIETRO=2 armate quando attacca

Aspettativa: stesso WR vs random in 1v1 di bot_euristico originale (~91%).
Verifica via test_bot_euristico_step.py.
"""

from __future__ import annotations
import random

import numpy as np

from .data import ADIACENZE
from .encoding import TERRITORIO_INDEX, INDEX_TERRITORIO
from .azioni import (
    INDICE_STOP_ATTACCO,
    INDICE_SKIP_SPOSTAMENTO,
)
from .bot_euristico import (
    SOGLIA_WIN_PROB,
    MIN_ARMATE_DIETRO,
    win_probability,
)

# Calibrato per riprodurre la media di 0-3 attacchi/turno del bot_euristico originale
# (media ~1.5 attacchi). Con prob_stop=0.40 a ogni call ATTACCO, attacchi attesi = 1/0.4 - 1 = 1.5.
PROB_STOP_ATTACCO = 0.40

# Coerente con bot_euristico originale (50/50 tris, 50/50 continua, 30% spostamento)
PROB_GIOCA_TRIS = 0.50
PROB_CONTINUA = 0.50
PROB_SPOSTA = 0.30


def bot_euristico_step(env, info: dict, rng: random.Random) -> int:
    """
    Decisione step-by-step: data un env e info, ritorna un singolo action idx
    seguendo le 3 regole euristiche di bot_euristico.

    Args:
        env: RisikoEnv (qualunque sotto-fase, tranne None).
        info: dict con 'action_mask' (np.ndarray bool di lunghezza 1765).
        rng: random.Random gia' inizializzato.

    Returns:
        int idx in [0, 1764] che e' legale per la sotto_fase corrente.
    """
    sotto_fase = env.sotto_fase
    mask = info["action_mask"]
    legali = np.where(mask)[0]

    # Casi degeneri
    if len(legali) == 0:
        return 0  # impossibile in pratica, fallback safe
    if len(legali) == 1:
        return int(legali[0])

    colore = env.bot_color  # giocatore che decide
    stato = env.stato

    # ─────────────────────────────────────────────────────────────
    # TRIS: 50% skip, 50% gioca random tra combinazioni
    # ─────────────────────────────────────────────────────────────
    if sotto_fase == "tris":
        # idx 0 = skip, idx 1..10 = combinazioni
        if rng.random() < PROB_GIOCA_TRIS:
            non_skip = legali[legali != 0]
            if len(non_skip) > 0:
                return int(rng.choice(non_skip))
        # default: skip (se legale, altrimenti random fra legali)
        return 0 if mask[0] else int(rng.choice(legali))

    # ─────────────────────────────────────────────────────────────
    # RINFORZO: regola 2 (60% confini, 40% uniforme)
    # idx 0..41 = territori
    # ─────────────────────────────────────────────────────────────
    if sotto_fase == "rinforzo":
        territori_propri = stato.territori_di(colore)
        if not territori_propri:
            return int(rng.choice(legali))  # impossibile in pratica

        confini = [
            t for t in territori_propri
            if any(stato.mappa[v].proprietario != colore for v in ADIACENZE[t])
        ]

        if confini and rng.random() < 0.60:
            scelta_terr = rng.choice(confini)
        else:
            scelta_terr = rng.choice(territori_propri)

        idx = TERRITORIO_INDEX[scelta_terr]
        # Validazione: idx deve essere legale (sanity)
        if mask[idx]:
            return idx
        return int(rng.choice(legali))  # fallback

    # ─────────────────────────────────────────────────────────────
    # ATTACCO: regole 1 e 3 + prob stop calibrata
    # idx 0..1763 = (da*42 + verso), idx 1764 = STOP
    # ─────────────────────────────────────────────────────────────
    if sotto_fase == "attacco":
        # Prob STOP a priori (simulazione "0-3 attacchi/turno" di bot_euristico)
        if rng.random() < PROB_STOP_ATTACCO and mask[INDICE_STOP_ATTACCO]:
            return INDICE_STOP_ATTACCO

        # Cerca candidati che rispettano regole 1 e 3
        candidati = []
        for idx in legali:
            if int(idx) == INDICE_STOP_ATTACCO:
                continue
            da_i = int(idx) // 42
            verso_i = int(idx) % 42
            da_t = INDEX_TERRITORIO[da_i]
            verso_t = INDEX_TERRITORIO[verso_i]
            armate_da = stato.mappa[da_t].armate
            armate_verso = stato.mappa[verso_t].armate

            # Regola 3: lascia almeno MIN_ARMATE_DIETRO armate dietro
            if armate_da < MIN_ARMATE_DIETRO + 1:
                continue
            armate_attacco = armate_da - MIN_ARMATE_DIETRO

            # Regola 1: win_prob >= soglia
            if win_probability(armate_attacco, armate_verso) < SOGLIA_WIN_PROB:
                continue

            candidati.append(int(idx))

        if candidati:
            return rng.choice(candidati)

        # Nessun candidato che rispetta le regole → STOP
        if mask[INDICE_STOP_ATTACCO]:
            return INDICE_STOP_ATTACCO
        return int(rng.choice(legali))  # impossibile in pratica

    # ─────────────────────────────────────────────────────────────
    # CONTINUA: 50% stop, 50% continua (come bot_random e bot_euristico)
    # idx 0 = stop, idx 1 = continua
    # ─────────────────────────────────────────────────────────────
    if sotto_fase == "continua":
        if rng.random() < PROB_CONTINUA:
            return 1 if mask[1] else int(legali[0])
        return 0 if mask[0] else int(legali[0])

    # ─────────────────────────────────────────────────────────────
    # QUANTITA_CONQUISTA: random fra le opzioni discrete (min/mid/max)
    # bot_random sceglie randint(minimo, massimo); qui idx 0/1/2 uniforme
    # ─────────────────────────────────────────────────────────────
    if sotto_fase == "quantita_conquista":
        return int(rng.choice(legali))

    # ─────────────────────────────────────────────────────────────
    # SPOSTAMENTO: 30% sposta (random da quelli legali), 70% skip
    # idx 0..1763 = (da*42 + verso), idx 1764 = SKIP
    # ─────────────────────────────────────────────────────────────
    if sotto_fase == "spostamento":
        if rng.random() < PROB_SPOSTA:
            non_skip = legali[legali != INDICE_SKIP_SPOSTAMENTO]
            if len(non_skip) > 0:
                return int(rng.choice(non_skip))
        return INDICE_SKIP_SPOSTAMENTO if mask[INDICE_SKIP_SPOSTAMENTO] else int(rng.choice(legali))

    # ─────────────────────────────────────────────────────────────
    # QUANTITA_SPOSTAMENTO: random fra le opzioni discrete (min/mid/max)
    # ─────────────────────────────────────────────────────────────
    if sotto_fase == "quantita_spostamento":
        return int(rng.choice(legali))

    # Fallback: sotto-fase inattesa → random fra legali
    return int(rng.choice(legali))
