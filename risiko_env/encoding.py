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

# === STAGE A2: opponent profile RIPROGETTATO ===
# Lezione da Stage A v1 (fallito, 19% WR vs baseline 29%):
# Le feature precedenti (aggressivita, focus, risk_tolerance, expansion_rate)
# erano semanticamente povere:
#   - dipendevano dal SUCCESSO degli attacchi (avversario aggressivo che fallisce
#     era indistinguibile da avversario passivo)
#   - risk_tolerance era una proxy fissa (0.4 sempre)
#   - 3 feature su 4 misuravano la stessa cosa (avversario ha conquistato qualcosa)
#
# Stage A2 usa FEATURE DI STATO (calcolate dallo stato corrente, non da storia):
# Per ogni avversario:
#   1. territori_norm     — # territori / 42 (forza territoriale)
#   2. armate_norm        — # armate / 130 (forza militare)
#   3. continenti_norm    — # continenti controllati / 6 (forza strategica)
#   4. confini_pov_norm   — # mie border-cells adiacenti a suoi territori / 42
#                            (vicinanza/minaccia geografica)
#   5. armate_confini_norm— armate sui suoi territori adiacenti ai miei /
#                            armate totali sue (concentrazione minaccia)
#   6. miei_minacciati_norm— # miei territori con un suo vicino piu' forte / # miei terr
#                            (minaccia immediata)
#   7. conquiste_recenti  — # territori netti guadagnati negli ultimi N turni / 5
#                            (espansione recente, FATTI non interpretazioni)
#   8. perdite_recenti    — # territori netti persi negli ultimi N turni / 5
#                            (vulnerabilita' recente)
#
# Feature 1-6: PURAMENTE DI STATO (deterministiche, calcolabili istantaneamente)
# Feature 7-8: STORICHE LEGGERE (ultimi N turni, contano fatti non eventi)
DIM_OPPONENT_PROFILE = (NUM_GIOCATORI - 1) * 8  # 24

# Finestra per feature 7-8 (conquiste/perdite recenti)
FINESTRA_OPPONENT_PROFILE = 5

# Stage A puo' essere disabilitato per retrocompatibilita' con modelli vecchi (318 feature)
# Variabile globale modificabile a runtime
STAGE_A_ATTIVO = True


def get_dim_observation() -> int:
    """Dimensione observation corrente (dipende da STAGE_A_ATTIVO)."""
    base = (
        DIM_MAPPA
        + DIM_OBIETTIVO_PROPRIO
        + DIM_CARTE_PROPRIE
        + DIM_AVVERSARI
        + DIM_CONTROLLO_CONTINENTI
        + DIM_FASE_E_TURNO
        + DIM_TRIS_GIOCATI
    )
    if STAGE_A_ATTIVO:
        base += DIM_OPPONENT_PROFILE
    return base


# Totale features observation (vale a import-time)
DIM_OBSERVATION = (
    DIM_MAPPA
    + DIM_OBIETTIVO_PROPRIO
    + DIM_CARTE_PROPRIE
    + DIM_AVVERSARI
    + DIM_CONTROLLO_CONTINENTI
    + DIM_FASE_E_TURNO
    + DIM_TRIS_GIOCATI
    + DIM_OPPONENT_PROFILE
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
    storia_mosse: dict | None = None,
) -> np.ndarray:
    """
    Codifica lo stato della partita dal punto di vista del giocatore `colore_pov`.

    Parametri:
        stato: stato corrente della partita
        colore_pov: di chi è la vista (es. "BLU"). Vede solo le info che gli spettano.
        fase_corrente: quale sotto-fase del turno è in corso (per encoding)
        storia_mosse: dict opzionale {colore: [mossa, ...]} dove ogni mossa è
            un dict con keys: 'turno', 'attaccato', 'attacchi_contro_pov',
            'territori_conquistati', 'ratio_medio'. Se None, opponent profile = 0.

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

    # 8. STAGE A: Opponent profile (ultime mosse di ogni avversario)
    # Solo se STAGE_A_ATTIVO=True (retrocompat con modelli vecchi)
    if STAGE_A_ATTIVO:
        parti.append(_codifica_opponent_profile(stato, colore_pov, storia_mosse))

    obs = np.concatenate(parti).astype(np.float32)
    dim_attesa = get_dim_observation()
    assert len(obs) == dim_attesa, (
        f"Dimensione observation errata: {len(obs)} vs atteso {dim_attesa}"
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


# ─────────────────────────────────────────────────────────────────────────
#  STAGE A2: OPPONENT PROFILE (feature di stato)
# ─────────────────────────────────────────────────────────────────────────

def _codifica_opponent_profile(
    stato: StatoPartita,
    colore_pov: str,
    storia_mosse: dict | None,
) -> np.ndarray:
    """
    Per ogni avversario, calcola 8 feature di profilo (Stage A2).

    Le prime 6 sono PURAMENTE DI STATO (dallo stato corrente, no storia).
    Le ultime 2 usano la storia recente per "conquiste/perdite ultimi N turni".

    1. territori_norm        — # territori / 42
    2. armate_norm           — # armate / 130
    3. continenti_norm       — # continenti controllati / 6
    4. confini_pov_norm      — # miei territori adiacenti a suoi / 42
    5. armate_confini_norm   — armate sue sui confini con me / armate sue totali
    6. miei_minacciati_norm  — # miei territori con un suo vicino piu' forte / # miei terr
    7. conquiste_recenti     — # territori netti guadagnati ultimi N turni / 5
    8. perdite_recenti       — # territori netti persi ultimi N turni / 5

    Output: np.ndarray shape (24,) = 8 feature × 3 avversari.
    """
    from .data import ADIACENZE, CONTINENTI as _CONT

    parti = []
    avversari = [c for c in COLORI_GIOCATORI if c != colore_pov]

    # Pre-calcola: insieme territori del POV e armate per territorio
    miei_territori = set()
    for t, st in stato.mappa.items():
        if st.proprietario == colore_pov:
            miei_territori.add(t)

    n_miei = max(1, len(miei_territori))  # evita div/0

    for avv in avversari:
        # === FEATURE DI STATO (1-6) ===

        # 1. Territori dell'avversario (norm su 42)
        suoi_territori = [t for t, s in stato.mappa.items() if s.proprietario == avv]
        territori_norm = len(suoi_territori) / float(NUM_TERRITORI)

        # 2. Armate dell'avversario (norm su 130 = cap)
        sue_armate_tot = sum(stato.mappa[t].armate for t in suoi_territori)
        armate_norm = min(1.0, sue_armate_tot / 130.0)

        # 3. Continenti controllati (norm su 6)
        n_continenti = 0
        for cont_terr in _CONT.values():
            if all(stato.mappa[t].proprietario == avv for t in cont_terr):
                n_continenti += 1
        continenti_norm = n_continenti / 6.0

        # 4. Confini con me: # MIEI territori adiacenti ad almeno un suo territorio
        #    (norm su 42 — quanto e' "vicino" geograficamente al mio impero)
        confini_count = 0
        suoi_set = set(suoi_territori)
        for mio_t in miei_territori:
            if any(vic in suoi_set for vic in ADIACENZE[mio_t]):
                confini_count += 1
        confini_pov_norm = confini_count / float(NUM_TERRITORI)

        # 5. Armate sue sui confini con me / armate sue totali
        #    (concentrazione minaccia: alta = ha massato truppe contro di me)
        armate_confini = 0
        miei_set = miei_territori
        for suo_t in suoi_territori:
            if any(vic in miei_set for vic in ADIACENZE[suo_t]):
                armate_confini += stato.mappa[suo_t].armate
        if sue_armate_tot > 0:
            armate_confini_norm = armate_confini / sue_armate_tot
        else:
            armate_confini_norm = 0.0

        # 6. Miei territori minacciati: # miei territori che hanno un vicino dell'avv
        #    con armate >= armate del mio territorio (norm su # miei terr)
        miei_minacciati = 0
        for mio_t in miei_territori:
            mie_armate_t = stato.mappa[mio_t].armate
            for vic in ADIACENZE[mio_t]:
                if vic in suoi_set and stato.mappa[vic].armate >= mie_armate_t:
                    miei_minacciati += 1
                    break  # 1 minaccia basta
        miei_minacciati_norm = miei_minacciati / float(n_miei)

        # === FEATURE DI STORIA RECENTE (7-8) ===
        # Usa la storia mosse per contare territori conquistati/persi ultimi N turni
        # NB: queste feature sono robuste perche' contano territori netti
        # (fatti, non interpretazioni di "attacchi").

        conquiste_recenti = 0.0
        perdite_recenti = 0.0
        if storia_mosse is not None and avv in storia_mosse and storia_mosse[avv]:
            mosse_recenti = list(storia_mosse[avv])[-FINESTRA_OPPONENT_PROFILE:]
            tot_conq = sum(m.get('territori_conquistati', 0) for m in mosse_recenti)
            tot_persi = sum(m.get('territori_persi', 0) for m in mosse_recenti)
            # Norm su 5 (territori in N turni)
            conquiste_recenti = min(1.0, tot_conq / 5.0)
            perdite_recenti = min(1.0, tot_persi / 5.0)

        parti.append(np.array(
            [
                territori_norm,
                armate_norm,
                continenti_norm,
                confini_pov_norm,
                armate_confini_norm,
                miei_minacciati_norm,
                conquiste_recenti,
                perdite_recenti,
            ],
            dtype=np.float32,
        ))

    return np.concatenate(parti)
