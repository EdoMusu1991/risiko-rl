"""
setup.py — Modulo 2: Setup di una partita.

Implementa le 4 fasi di setup descritte nella specifica sezione 3:
- Fase A: distribuzione territori (con vincolo continentale)
- Fase B: distribuzione obiettivi segreti
- Fase C: piazzamento iniziale (7 round di 3-2-1 carri)
- Fase D: setup mazzo (mescolamento)

Output: uno StatoPartita pronto per il primo turno di Blu, round 1.

Per riproducibilità tutte le funzioni accettano un random.Random instance.
NON si usa il modulo random globale.

Specifica di riferimento: risiko_specifica_v1.2.md sezione 3.
"""

import random
from typing import Optional

from .data import (
    TUTTI_TERRITORI,
    CONTINENTI,
    COLORI_GIOCATORI,
    OBIETTIVI,
    TERRITORI_PER_GIOCATORE,
    CARRI_PIAZZAMENTO_INIZIALE,
    limite_continente_distribuzione,
)
from .stato import (
    StatoPartita,
    crea_mazzo_completo,
)


# ─────────────────────────────────────────────────────────────────────────
#  FASE A — DISTRIBUZIONE TERRITORI
# ─────────────────────────────────────────────────────────────────────────

def distribuisci_territori(stato: StatoPartita, rng: random.Random) -> None:
    """
    Distribuisce i 42 territori tra i 4 giocatori rispettando il vincolo
    continentale (nessuno può avere più della metà dei territori di un continente).

    Modifica `stato.mappa` in-place: assegna proprietario e mette 1 carro
    su ogni territorio.

    Algoritmo: due passate.
    1) Passata greedy: per ogni giocatore in ordine, assegna territori dal pool
       mescolato saltando quelli che violerebbero il vincolo.
    2) Passata fallback: i territori rimasti (che violano il cap) vengono
       distribuiti al giocatore con meno territori. Il vincolo è quindi
       "tentato" non "garantito", come specificato nella sezione 3.1 punto 4.
    """
    # Pool mescolato di tutti i 42 territori
    pool = list(TUTTI_TERRITORI)
    rng.shuffle(pool)

    quote = {col: TERRITORI_PER_GIOCATORE[col] for col in COLORI_GIOCATORI}
    assegnati: dict[str, list[str]] = {col: [] for col in COLORI_GIOCATORI}

    # ── PASSATA 1: greedy con vincolo ──────────────────────────────
    # Per ogni giocatore, scorri il pool e prendi quelli che non violano il cap.
    # I territori che violerebbero restano nel pool per il giocatore successivo.
    for colore in COLORI_GIOCATORI:
        target = quote[colore]
        nuovo_pool = []  # territori non presi da questo giocatore (per il prossimo)

        for t in pool:
            if len(assegnati[colore]) < target and not _viola_cap_continente(t, assegnati[colore]):
                assegnati[colore].append(t)
            else:
                nuovo_pool.append(t)

        pool = nuovo_pool  # passa al prossimo giocatore solo i non assegnati

    # ── PASSATA 2: fallback per territori rimasti ───────────────────
    # I territori ancora nel pool sono quelli che violavano il cap per tutti
    # i giocatori che li hanno visti. Li distribuiamo cercando comunque di
    # rispettare le quote (e ignorando il cap come ultima ratio).
    for t in pool:
        # Trova i giocatori che non hanno ancora raggiunto la quota
        sotto_quota = [c for c in COLORI_GIOCATORI if len(assegnati[c]) < quote[c]]
        if not sotto_quota:
            # Caso limite: tutte le quote raggiunte ma rimangono territori (impossibile
            # con 42 territori e quote 11+11+10+10=42, ma per sicurezza)
            scelto = min(COLORI_GIOCATORI, key=lambda c: len(assegnati[c]))
        else:
            # Tra i sotto-quota, prendi quello con meno territori (più equo)
            scelto = min(sotto_quota, key=lambda c: len(assegnati[c]))
        assegnati[scelto].append(t)

    # ── Applica le assegnazioni alla mappa ────────────────────────
    for colore, territori in assegnati.items():
        for t in territori:
            stato.mappa[t].proprietario = colore
            stato.mappa[t].armate = 1


def _viola_cap_continente(territorio: str, gia_assegnati: list[str]) -> bool:
    """
    Restituisce True se aggiungere `territorio` a `gia_assegnati` violerebbe
    il vincolo "max metà del continente".
    """
    for cont, terrs_cont in CONTINENTI.items():
        if territorio not in terrs_cont:
            continue
        # Quanti territori di questo continente ha già il giocatore
        gia_in_cont = sum(1 for t in gia_assegnati if t in terrs_cont)
        if gia_in_cont + 1 > limite_continente_distribuzione(cont):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
#  FASE B — DISTRIBUZIONE OBIETTIVI
# ─────────────────────────────────────────────────────────────────────────

def distribuisci_obiettivi(stato: StatoPartita, rng: random.Random) -> None:
    """
    Assegna 1 obiettivo segreto a ognuno dei 4 giocatori.
    Pesca a caso 4 obiettivi diversi tra i 16 disponibili.
    """
    obiettivi_ids = list(OBIETTIVI.keys())
    rng.shuffle(obiettivi_ids)
    pescati = obiettivi_ids[:4]

    for colore, obj_id in zip(COLORI_GIOCATORI, pescati):
        stato.giocatori[colore].obiettivo_id = obj_id


# ─────────────────────────────────────────────────────────────────────────
#  FASE C — PIAZZAMENTO INIZIALE
# ─────────────────────────────────────────────────────────────────────────

def piazzamento_iniziale_random(stato: StatoPartita, rng: random.Random) -> None:
    """
    Esegue il piazzamento iniziale dei carri (7 round di 3-2-1 carri per
    giocatore secondo specifica 3.3).

    Strategia: distribuzione casuale sui propri territori.
    Questa è una strategia PLACEHOLDER — quando il bot RL sarà pronto,
    sarà lui a decidere dove piazzare nei suoi turni.

    Modifica stato.mappa in-place aggiungendo i carri.
    """
    for round_idx in range(7):  # 7 round di piazzamento
        for colore in COLORI_GIOCATORI:
            num_carri = CARRI_PIAZZAMENTO_INIZIALE[colore][round_idx]
            territori_propri = stato.territori_di(colore)

            # Piazza casualmente sui propri territori
            for _ in range(num_carri):
                t = rng.choice(territori_propri)
                stato.mappa[t].armate += 1


# ─────────────────────────────────────────────────────────────────────────
#  FASE D — SETUP MAZZO
# ─────────────────────────────────────────────────────────────────────────

def setup_mazzo(stato: StatoPartita, rng: random.Random) -> None:
    """
    Crea il mazzo completo (44 carte), lo mescola, lo assegna a stato.mazzo_attivo.
    Pila scarti vuota all'inizio.
    """
    mazzo = crea_mazzo_completo()
    rng.shuffle(mazzo)
    stato.mazzo_attivo = mazzo
    stato.pila_scarti = []


# ─────────────────────────────────────────────────────────────────────────
#  FUNZIONE INTEGRATIVA: setup completo
# ─────────────────────────────────────────────────────────────────────────

def crea_partita_iniziale(seed: Optional[int] = None) -> StatoPartita:
    """
    Crea una partita pronta a iniziare: tutte le 4 fasi di setup completate.

    Parametri:
        seed: se fornito, garantisce riproducibilità della partita.
              Stesso seed → stessa distribuzione, stessi obiettivi, stesso mazzo.
              Default: None = casualità non riproducibile.

    Restituisce uno StatoPartita con:
    - 42 territori assegnati ai 4 giocatori (11/11/10/10) con armate piazzate
    - 1 obiettivo segreto per giocatore
    - Mazzo mescolato e pronto, pila scarti vuota
    - round_corrente=1, giocatore_corrente="BLU" (pronto per il primo turno)
    """
    rng = random.Random(seed)
    stato = StatoPartita()

    # Fase A: distribuzione territori
    distribuisci_territori(stato, rng)

    # Fase B: obiettivi (PRIMA del piazzamento, vedi specifica 3.2)
    distribuisci_obiettivi(stato, rng)

    # Fase C: piazzamento iniziale
    piazzamento_iniziale_random(stato, rng)

    # Fase D: mazzo
    setup_mazzo(stato, rng)

    # Pronto per il primo turno
    stato.round_corrente = 1
    stato.giocatore_corrente = "BLU"
    stato.conquiste_turno_corrente = {col: 0 for col in COLORI_GIOCATORI}

    return stato
