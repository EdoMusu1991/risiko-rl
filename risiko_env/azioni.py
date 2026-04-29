"""
azioni.py — Modulo 5b: Action space e action masking.

Definisce le azioni che il bot RL può prendere durante una partita.
Ogni "tipo di decisione" ha il suo action space e la sua maschera di legalità.

I 5 tipi di decisione:
1. AZIONE_TRIS — quali tris giocare (0, 1, o 2)
2. AZIONE_RINFORZO — su quale territorio piazzare la prossima armata
3. AZIONE_ATTACCO — quale attacco fare (da, verso) o stop
4. AZIONE_CONTINUA — continuare un combattimento o fermarsi
5. AZIONE_SPOSTAMENTO — fare uno spostamento finale (da, verso, qty) o skip

Per ognuna: una funzione che genera lo space, una che genera la mask.
"""

import numpy as np

from .data import (
    TUTTI_TERRITORI,
    ADIACENZE,
    COLORI_GIOCATORI,
    FANTE,
    CANNONE,
    CAVALLO,
    JOLLY,
    BONUS_TRIS_3_UGUALI,
    BONUS_TRIS_3_DIVERSI,
    BONUS_TRIS_JOLLY_PIU_2,
)
from .stato import StatoPartita, Carta
from .encoding import TERRITORIO_INDEX, INDEX_TERRITORIO


# ═════════════════════════════════════════════════════════════════════════
#  COSTANTI ACTION SPACE
# ═════════════════════════════════════════════════════════════════════════

# AZIONE_TRIS: max 11 opzioni (0=skip, 1-10=combinazioni di tris enumerate)
# Calcolo: con 7 carte massime ci sono al più ~10 combinazioni di 1-2 tris.
# 0 = non gioco tris
# 1-10 = giochi una combinazione specifica enumerata da enumera_combinazioni_tris()
NUM_AZIONI_TRIS = 11  # 0=skip + 10 combinazioni max

# AZIONE_RINFORZO: 42 opzioni (1 per ogni territorio)
NUM_AZIONI_RINFORZO = 42

# AZIONE_ATTACCO: 42×42 = 1764 coppie (da, verso) + 1 stop = 1765
NUM_AZIONI_ATTACCO = 42 * 42 + 1
INDICE_STOP_ATTACCO = 42 * 42  # ultimo indice riservato per "stop attacchi"

# AZIONE_CONTINUA: 2 opzioni (0=stop, 1=continua)
NUM_AZIONI_CONTINUA = 2

# AZIONE_SPOSTAMENTO_QUANTITA: 3 opzioni discrete dopo conquista
# 0 = minimo (= num dadi ultimo lancio)
# 1 = intermedio (~ metà tra min e max)
# 2 = massimo (= armate_origine - 1)
NUM_AZIONI_QUANTITA = 3

# AZIONE_SPOSTAMENTO_FINALE: 42×42 = 1764 coppie + 1 skip = 1765
NUM_AZIONI_SPOSTAMENTO = 42 * 42 + 1
INDICE_SKIP_SPOSTAMENTO = 42 * 42


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE TRIS
# ═════════════════════════════════════════════════════════════════════════

def enumera_combinazioni_tris(carte: list[Carta]) -> list[list[tuple[list[Carta], int]]]:
    """
    Enumera fino a 10 combinazioni distinte di giocate-tris.

    Ogni combinazione è una lista di tris (0, 1, o 2 tris) che il giocatore
    sceglie di giocare nello stesso turno.

    Restituisce una lista ordinata. L'indice 0 è SEMPRE "non giocare nulla".
    Gli indici 1+ sono combinazioni alternative.

    Esempio: con 6 carte (3 fanti + 3 cannoni) ci sono opzioni:
    - 0: non gioco
    - 1: gioco solo i 3 fanti
    - 2: gioco solo i 3 cannoni
    - 3: gioco entrambi (2 tris contemporaneamente)
    """
    combinazioni: list[list[tuple[list[Carta], int]]] = [[]]  # 0 = niente

    if len(carte) < 3:
        return combinazioni

    # Trova tutti i tris singoli possibili
    tris_singoli = _trova_tris_singoli(carte)

    # Aggiungi ogni tris singolo come combinazione
    for tris in tris_singoli:
        combinazioni.append([tris])
        if len(combinazioni) >= NUM_AZIONI_TRIS:
            break

    # Aggiungi combinazioni di 2 tris disgiunti (se possibile)
    if len(combinazioni) < NUM_AZIONI_TRIS and len(carte) >= 6:
        for i, tris1 in enumerate(tris_singoli):
            for tris2 in tris_singoli[i+1:]:
                # Verifica che siano disgiunti (nessuna carta in comune)
                set1 = set(id(c) for c in tris1[0])
                set2 = set(id(c) for c in tris2[0])
                if not (set1 & set2):
                    combinazioni.append([tris1, tris2])
                    if len(combinazioni) >= NUM_AZIONI_TRIS:
                        break
            if len(combinazioni) >= NUM_AZIONI_TRIS:
                break

    return combinazioni


def _trova_tris_singoli(carte: list[Carta]) -> list[tuple[list[Carta], int]]:
    """
    Trova tutti i possibili tris singoli dalle carte.
    Restituisce lista di (carte_del_tris, bonus_armate).
    """
    tris_trovati = []

    fanti = [c for c in carte if c.simbolo == FANTE]
    cannoni = [c for c in carte if c.simbolo == CANNONE]
    cavalli = [c for c in carte if c.simbolo == CAVALLO]
    jolly = [c for c in carte if c.is_jolly]

    # 3 uguali
    if len(fanti) >= 3:
        tris_trovati.append((fanti[:3], BONUS_TRIS_3_UGUALI))
    if len(cannoni) >= 3:
        tris_trovati.append((cannoni[:3], BONUS_TRIS_3_UGUALI))
    if len(cavalli) >= 3:
        tris_trovati.append((cavalli[:3], BONUS_TRIS_3_UGUALI))

    # 3 diversi
    if fanti and cannoni and cavalli:
        tris_trovati.append((
            [fanti[0], cannoni[0], cavalli[0]],
            BONUS_TRIS_3_DIVERSI,
        ))

    # Jolly + 2 uguali
    for j in jolly:
        if len(fanti) >= 2:
            tris_trovati.append((
                [j, fanti[0], fanti[1]],
                BONUS_TRIS_JOLLY_PIU_2,
            ))
        if len(cannoni) >= 2:
            tris_trovati.append((
                [j, cannoni[0], cannoni[1]],
                BONUS_TRIS_JOLLY_PIU_2,
            ))
        if len(cavalli) >= 2:
            tris_trovati.append((
                [j, cavalli[0], cavalli[1]],
                BONUS_TRIS_JOLLY_PIU_2,
            ))

    return tris_trovati


def maschera_tris(combinazioni: list) -> np.ndarray:
    """
    Maschera booleana per AZIONE_TRIS.
    True per gli indici corrispondenti a combinazioni effettivamente disponibili.
    """
    mask = np.zeros(NUM_AZIONI_TRIS, dtype=bool)
    for i in range(min(len(combinazioni), NUM_AZIONI_TRIS)):
        mask[i] = True
    return mask


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE RINFORZO
# ═════════════════════════════════════════════════════════════════════════

def maschera_rinforzo(stato: StatoPartita, colore: str) -> np.ndarray:
    """
    Maschera per AZIONE_RINFORZO: quali territori posso scegliere.
    Solo i territori posseduti dal giocatore.
    """
    mask = np.zeros(NUM_AZIONI_RINFORZO, dtype=bool)
    for t in stato.territori_di(colore):
        mask[TERRITORIO_INDEX[t]] = True
    return mask


def decodifica_azione_rinforzo(azione: int) -> str:
    """Indice → nome territorio."""
    return INDEX_TERRITORIO[azione]


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE ATTACCO
# ═════════════════════════════════════════════════════════════════════════

def codifica_attacco(da: str, verso: str) -> int:
    """Codifica una coppia (da, verso) in un singolo indice 0-1763."""
    return TERRITORIO_INDEX[da] * 42 + TERRITORIO_INDEX[verso]


def decodifica_azione_attacco(azione: int) -> tuple[str, str] | None:
    """
    Decodifica un indice di azione attacco.
    Restituisce (da, verso) o None se l'indice è "stop attacchi".
    """
    if azione == INDICE_STOP_ATTACCO:
        return None  # stop
    da_idx = azione // 42
    verso_idx = azione % 42
    return (INDEX_TERRITORIO[da_idx], INDEX_TERRITORIO[verso_idx])


def maschera_attacco(stato: StatoPartita, colore: str) -> np.ndarray:
    """
    Maschera per AZIONE_ATTACCO: quali coppie (da, verso) sono attacchi legali.
    Include sempre l'azione "stop" (ultimo indice).

    Un attacco è legale se:
    - `da` è del colore
    - `verso` è di un avversario
    - sono adiacenti
    - `da` ha almeno 2 armate
    """
    mask = np.zeros(NUM_AZIONI_ATTACCO, dtype=bool)

    territori_propri = stato.territori_di(colore)
    for da in territori_propri:
        sd = stato.mappa[da]
        if sd.armate < 2:
            continue
        for verso in ADIACENZE[da]:
            sv = stato.mappa[verso]
            if sv.proprietario in (None, colore):
                continue
            idx = codifica_attacco(da, verso)
            mask[idx] = True

    # "Stop" è sempre un'opzione legale
    mask[INDICE_STOP_ATTACCO] = True

    return mask


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE CONTINUA-COMBATTIMENTO
# ═════════════════════════════════════════════════════════════════════════

def maschera_continua(stato: StatoPartita, colore: str,
                      da: str, verso: str) -> np.ndarray:
    """
    Maschera per AZIONE_CONTINUA: 0=stop, 1=continua.
    Stop è sempre legale. Continua è legale solo se:
    - L'attaccante ha ancora >= 2 armate
    - Il difensore ha ancora >= 1 armata (cioè non l'ho già conquistato)
    - I territori esistono e hanno la stessa proprietà di prima
    """
    mask = np.array([True, False], dtype=bool)  # stop sempre legale

    sd = stato.mappa.get(da)
    sv = stato.mappa.get(verso)
    if sd is None or sv is None:
        return mask

    if sd.proprietario != colore:
        return mask
    if sv.proprietario in (None, colore):
        return mask  # già conquistato o vuoto
    if sd.armate < 2 or sv.armate < 1:
        return mask
    if verso not in ADIACENZE[da]:
        return mask

    mask[1] = True
    return mask


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE QUANTITA DA SPOSTARE (post-conquista)
# ═════════════════════════════════════════════════════════════════════════

def calcola_quantita_da_azione(
    azione: int,
    minimo: int,
    massimo: int,
) -> int:
    """
    Converte l'azione discreta (0=min, 1=intermedio, 2=max) in quantità reale.
    Se min == max, ritorna sempre min.
    """
    if azione == 0:
        return minimo
    if azione == 2:
        return massimo
    # Intermedio
    return (minimo + massimo) // 2


def maschera_quantita(minimo: int, massimo: int) -> np.ndarray:
    """
    Tutte e 3 le opzioni sempre legali se min <= max.
    Se min == max, solo l'opzione 0 è legale (le altre coincidono).
    """
    mask = np.array([True, True, True], dtype=bool)
    if minimo == massimo:
        mask[1] = False
        mask[2] = False
    elif minimo + 1 == massimo:
        # solo min e max distinti, intermedio = min
        mask[1] = False
    return mask


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE SPOSTAMENTO FINALE
# ═════════════════════════════════════════════════════════════════════════

def maschera_spostamento(stato: StatoPartita, colore: str) -> np.ndarray:
    """
    Maschera per AZIONE_SPOSTAMENTO: quali coppie (da, verso) sono legali.
    Include sempre "skip" (ultimo indice).

    Uno spostamento è legale se:
    - da e verso sono entrambi del colore
    - sono adiacenti
    - quantità minima 1 può essere effettivamente spostata
      (cioè da ha abbastanza armate per soddisfare il minimo da lasciare + 1)
    """
    mask = np.zeros(NUM_AZIONI_SPOSTAMENTO, dtype=bool)

    territori_propri = stato.territori_di(colore)
    for da in territori_propri:
        sd = stato.mappa[da]
        # Calcola minimo da lasciare (1 in nicchia, 2 altrimenti)
        confinanti = ADIACENZE[da]
        in_nicchia = all(
            stato.mappa[c].proprietario == colore for c in confinanti
        )
        min_da_lasciare = 1 if in_nicchia else 2
        max_spostabile = sd.armate - min_da_lasciare
        if max_spostabile < 1:
            continue

        # Per ogni adiacente proprio, è uno spostamento valido
        for verso in confinanti:
            sv = stato.mappa[verso]
            if sv.proprietario != colore:
                continue
            idx = codifica_attacco(da, verso)  # stesso schema di codifica
            mask[idx] = True

    # Skip è sempre legale
    mask[INDICE_SKIP_SPOSTAMENTO] = True
    return mask


def decodifica_azione_spostamento(azione: int) -> tuple[str, str] | None:
    """Decodifica indice spostamento. None se skip."""
    if azione == INDICE_SKIP_SPOSTAMENTO:
        return None
    da_idx = azione // 42
    verso_idx = azione % 42
    return (INDEX_TERRITORIO[da_idx], INDEX_TERRITORIO[verso_idx])


# ═════════════════════════════════════════════════════════════════════════
#  HELPER: dimensioni totali (per Gymnasium MultiDiscrete in futuro)
# ═════════════════════════════════════════════════════════════════════════

# Esposizione delle dimensioni per il modulo 5c (env Gymnasium)
DIMENSIONI_AZIONI = {
    "tris": NUM_AZIONI_TRIS,
    "rinforzo": NUM_AZIONI_RINFORZO,
    "attacco": NUM_AZIONI_ATTACCO,
    "continua": NUM_AZIONI_CONTINUA,
    "quantita": NUM_AZIONI_QUANTITA,
    "spostamento": NUM_AZIONI_SPOSTAMENTO,
}
