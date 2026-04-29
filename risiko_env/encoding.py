"""
encoding.py — Modulo 5a: Encoding dell'observation.

Trasforma lo StatoPartita in un vettore numerico di dimensione fissa
che la rete neurale del bot può elaborare.

Il vettore include solo le informazioni che il bot ha il diritto di vedere
(specifica 8): mappa, propri carte/obiettivo, statistiche pubbliche degli
avversari. NON include obiettivi avversari né le loro carte specifiche.

Specifica di riferimento: risiko_specifica_v1.2.md sezione 8.
"""

import numpy as np

from .data import (
    TUTTI_TERRITORI,
    COLORI_GIOCATORI,
    OBIETTIVI,
    CONTINENTI,
    BONUS_CONTINENTE,
    FANTE,
    CANNONE,
    CAVALLO,
    JOLLY,
    MAX_ARMATE_TOTALI,
    MAX_CARTE_MANO,
)
from .stato import StatoPartita


# ─────────────────────────────────────────────────────────────────────────
#  COSTANTI ENCODING
# ─────────────────────────────────────────────────────────────────────────

NUM_TERRITORI = 42
NUM_CONTINENTI = 6
NUM_OBIETTIVI = 16
NUM_GIOCATORI = 4

# Dimensioni delle sezioni dell'observation:
DIM_MAPPA = NUM_TERRITORI * 6  # 42 territori × (4 one-hot proprietario + armate_norm + è_in_obj)
DIM_OBIETTIVO_PROPRIO = NUM_OBIETTIVI  # one-hot dell'obiettivo proprio
DIM_CARTE_PROPRIE = 5  # [num_fanti, num_cannoni, num_cavalli, num_jolly, totale]
DIM_AVVERSARI = (NUM_GIOCATORI - 1) * 4  # per ogni avversario: [num_terr, num_armate, num_carte, vivo]
DIM_CONTROLLO_CONTINENTI = NUM_CONTINENTI * NUM_GIOCATORI  # chi controlla ogni continente
DIM_FASE_E_TURNO = 6  # [round_norm, fase_one_hot(4), conquiste_turno_norm]
DIM_TRIS_GIOCATI = 3  # numero di tris giocati nella partita per ogni simbolo (info pubblica)

# Totale features observation
DIM_OBSERVATION = (
    DIM_MAPPA
    + DIM_OBIETTIVO_PROPRIO
    + DIM_CARTE_PROPRIE
    + DIM_AVVERSARI
    + DIM_CONTROLLO_CONTINENTI
    + DIM_FASE_E_TURNO
    + DIM_TRIS_GIOCATI
)


# Ordine canonico territori (assegna un indice da 0 a 41)
TERRITORIO_INDEX: dict[str, int] = {t: i for i, t in enumerate(TUTTI_TERRITORI)}
INDEX_TERRITORIO: dict[int, str] = {i: t for t, i in TERRITORIO_INDEX.items()}

# Ordine canonico colori (0=BLU, 1=ROSSO, 2=VERDE, 3=GIALLO)
COLORE_INDEX: dict[str, int] = {c: i for i, c in enumerate(COLORI_GIOCATORI)}
INDEX_COLORE: dict[int, str] = {i: c for c, i in COLORE_INDEX.items()}

# Ordine canonico continenti (per encoding)
CONTINENTE_INDEX: dict[str, int] = {c: i for i, c in enumerate(CONTINENTI.keys())}

# Ordine canonico fasi (per encoding fase corrente)
FASI_ORDINE = ["TRIS_E_RINFORZI", "ATTACCHI", "SPOSTAMENTO", "PESCA_CARTA"]
FASE_INDEX: dict[str, int] = {f: i for i, f in enumerate(FASI_ORDINE)}


# ─────────────────────────────────────────────────────────────────────────
#  ENCODING PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────

def codifica_osservazione(
    stato: StatoPartita,
    colore_pov: str,
    fase_corrente: str = "TRIS_E_RINFORZI",
) -> np.ndarray:
    """
    Codifica lo stato della partita dal punto di vista del giocatore `colore_pov`.

    Parametri:
        stato: stato corrente della partita
        colore_pov: di chi è la vista (es. "BLU"). Vede solo le info che gli spettano.
        fase_corrente: quale sotto-fase del turno è in corso (per encoding)

    Restituisce un np.ndarray di dimensione DIM_OBSERVATION (float32).
    """
    parti = []

    # 1. Mappa: 42 territori × 6 features
    parti.append(_codifica_mappa(stato, colore_pov))

    # 2. Obiettivo proprio (one-hot)
    parti.append(_codifica_obiettivo_proprio(stato, colore_pov))

    # 3. Carte proprie (conteggio per simbolo)
    parti.append(_codifica_carte_proprie(stato, colore_pov))

    # 4. Statistiche avversari pubbliche
    parti.append(_codifica_avversari(stato, colore_pov))

    # 5. Controllo continenti (chi possiede ogni continente intero)
    parti.append(_codifica_controllo_continenti(stato))

    # 6. Fase corrente, round, conquiste
    parti.append(_codifica_fase_e_turno(stato, colore_pov, fase_corrente))

    # 7. Tris pubblicamente giocati nella partita (info da pila scarti)
    parti.append(_codifica_tris_giocati(stato))

    obs = np.concatenate(parti).astype(np.float32)
    assert len(obs) == DIM_OBSERVATION, (
        f"Dimensione observation errata: {len(obs)} vs atteso {DIM_OBSERVATION}"
    )
    return obs


# ─────────────────────────────────────────────────────────────────────────
#  ENCODING SEZIONI
# ─────────────────────────────────────────────────────────────────────────

def _codifica_mappa(stato: StatoPartita, colore_pov: str) -> np.ndarray:
    """
    Per ogni territorio, 6 features:
    - 4 one-hot per il proprietario (BLU, ROSSO, VERDE, GIALLO)
    - 1 valore: armate normalizzate (armate / 30, cap a 1.0)
    - 1 valore: 1 se il territorio è nell'obiettivo del POV, 0 altrimenti

    Questo dà al bot una rappresentazione spaziale completa della mappa.
    """
    obj_id = stato.giocatori[colore_pov].obiettivo_id
    territori_in_obj = (
        OBIETTIVI[obj_id]["territori"] if obj_id is not None else frozenset()
    )

    features = np.zeros((NUM_TERRITORI, 6), dtype=np.float32)

    for t, idx in TERRITORIO_INDEX.items():
        s = stato.mappa[t]
        # One-hot proprietario
        if s.proprietario in COLORE_INDEX:
            features[idx, COLORE_INDEX[s.proprietario]] = 1.0
        # Armate normalizzate (cap a 30 per stabilità numerica)
        features[idx, 4] = min(s.armate / 30.0, 1.0)
        # Flag in obiettivo
        features[idx, 5] = 1.0 if t in territori_in_obj else 0.0

    return features.flatten()


def _codifica_obiettivo_proprio(stato: StatoPartita, colore_pov: str) -> np.ndarray:
    """One-hot dell'obiettivo del giocatore (16 dimensioni)."""
    vec = np.zeros(NUM_OBIETTIVI, dtype=np.float32)
    obj_id = stato.giocatori[colore_pov].obiettivo_id
    if obj_id is not None and 1 <= obj_id <= NUM_OBIETTIVI:
        vec[obj_id - 1] = 1.0
    return vec


def _codifica_carte_proprie(stato: StatoPartita, colore_pov: str) -> np.ndarray:
    """
    Conteggio delle carte proprie per simbolo:
    [num_fanti, num_cannoni, num_cavalli, num_jolly, totale_normalizzato]
    """
    giocatore = stato.giocatori[colore_pov]
    n_fanti = sum(1 for c in giocatore.carte if c.simbolo == FANTE)
    n_cannoni = sum(1 for c in giocatore.carte if c.simbolo == CANNONE)
    n_cavalli = sum(1 for c in giocatore.carte if c.simbolo == CAVALLO)
    n_jolly = sum(1 for c in giocatore.carte if c.simbolo == JOLLY)
    totale = giocatore.num_carte() / MAX_CARTE_MANO  # normalizzato 0-1

    return np.array([n_fanti, n_cannoni, n_cavalli, n_jolly, totale],
                    dtype=np.float32)


def _codifica_avversari(stato: StatoPartita, colore_pov: str) -> np.ndarray:
    """
    Per ogni avversario (3 totali, in ordine canonico saltando il POV):
    - num_territori normalizzato (/ 42)
    - num_armate normalizzato (/ 130)
    - num_carte normalizzato (/ 7)
    - vivo (0 o 1)

    Totale: 3 × 4 = 12 features.
    """
    avversari = [c for c in COLORI_GIOCATORI if c != colore_pov]
    features = []
    for col in avversari:
        n_terr = stato.num_territori_di(col)
        n_armate = stato.num_armate_di(col)
        n_carte = stato.giocatori[col].num_carte()
        vivo = 1.0 if stato.giocatori[col].vivo else 0.0

        features.extend([
            n_terr / NUM_TERRITORI,
            n_armate / MAX_ARMATE_TOTALI,
            n_carte / MAX_CARTE_MANO,
            vivo,
        ])

    return np.array(features, dtype=np.float32)


def _codifica_controllo_continenti(stato: StatoPartita) -> np.ndarray:
    """
    Per ogni continente, 4 valori one-hot indicando chi lo controlla totalmente.
    Se nessuno controlla un continente, tutti zero per quel continente.

    Totale: 6 continenti × 4 colori = 24 features.
    """
    features = np.zeros((NUM_CONTINENTI, NUM_GIOCATORI), dtype=np.float32)

    for cont, terrs in CONTINENTI.items():
        cont_idx = CONTINENTE_INDEX[cont]
        # Trova chi possiede tutti i territori del continente
        proprietari = set(
            stato.mappa[t].proprietario
            for t in terrs
            if stato.mappa[t].proprietario is not None
        )
        if len(proprietari) == 1:
            unico = list(proprietari)[0]
            if unico in COLORE_INDEX:
                features[cont_idx, COLORE_INDEX[unico]] = 1.0

    return features.flatten()


def _codifica_fase_e_turno(
    stato: StatoPartita,
    colore_pov: str,
    fase_corrente: str,
) -> np.ndarray:
    """
    Features della fase corrente:
    - round_corrente normalizzato (/ 60)
    - 4 one-hot della fase
    - conquiste_turno_corrente normalizzate (/ 5)

    Totale: 6 features.
    """
    round_norm = stato.round_corrente / 60.0
    fase_oh = np.zeros(4, dtype=np.float32)
    if fase_corrente in FASE_INDEX:
        fase_oh[FASE_INDEX[fase_corrente]] = 1.0
    conquiste = stato.conquiste_turno_corrente.get(colore_pov, 0)
    conquiste_norm = min(conquiste / 5.0, 1.0)  # cap a 5 per stabilità

    return np.array([round_norm, *fase_oh, conquiste_norm], dtype=np.float32)


def _codifica_tris_giocati(stato: StatoPartita) -> np.ndarray:
    """
    Numero totale di carte di ciascun simbolo finite nella pila scarti.
    Info pubblica (specifica 5.5: i tris giocati sono pubblici).

    Totale: 3 features (fanti, cannoni, cavalli scartati).
    Normalizzato per il numero massimo possibile (14 per simbolo).
    """
    n_fanti = sum(1 for c in stato.pila_scarti if c.simbolo == FANTE) / 14.0
    n_cannoni = sum(1 for c in stato.pila_scarti if c.simbolo == CANNONE) / 14.0
    n_cavalli = sum(1 for c in stato.pila_scarti if c.simbolo == CAVALLO) / 14.0
    return np.array([n_fanti, n_cannoni, n_cavalli], dtype=np.float32)
