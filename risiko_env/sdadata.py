"""
sdadata.py — Modulo 4: Sdadata e fine partita.

Implementa il meccanismo di terminazione della partita:
- Sdadata obbligatoria dal round 35 (Giallo) / 36 (Blu, Rosso, Verde)
- Lancio di 2 dadi: somma <= soglia → partita finita
- Cap di sicurezza al round 60
- Determinazione vincitore via cascata criteri (delegata a obiettivi.determina_vincitore)

Specifica di riferimento: risiko_specifica_v1.2.md sezioni 7.1.B, 7.2, 7.3.
"""

import random

from .data import soglia_sdadata, ROUND_CAP_SICUREZZA
from .stato import StatoPartita
from .obiettivi import determina_vincitore


# ─────────────────────────────────────────────────────────────────────────
#  CONDIZIONI PER SDADATA
# ─────────────────────────────────────────────────────────────────────────

def deve_tirare_sdadata(stato: StatoPartita, colore: str) -> bool:
    """
    Verifica se il giocatore `colore` deve OBBLIGATORIAMENTE tirare la sdadata
    a fine turno (specifica 7.2.3).

    Condizioni:
    1. È nel round in cui può tirare (≥35 Giallo, ≥36 altri)
    2. Ha conquistato ≤ 2 territori nel turno corrente

    Note: la sdadata è OBBLIGATORIA, non opzionale. L'unico modo per non tirarla
    è conquistare 3 o più territori nel turno.
    """
    soglia = soglia_sdadata(colore, stato.round_corrente)
    if soglia is None:
        return False  # Round troppo presto

    conquiste = stato.conquiste_turno_corrente.get(colore, 0)
    if conquiste > 2:
        return False  # Conquistato 3+ → niente sdadata

    return True


# ─────────────────────────────────────────────────────────────────────────
#  TIRO DELLA SDADATA
# ─────────────────────────────────────────────────────────────────────────

def tira_sdadata(
    stato: StatoPartita,
    colore: str,
    rng: random.Random,
) -> tuple[bool, int, int, int]:
    """
    Esegue il tiro di 2 dadi della sdadata.

    Restituisce una tupla:
        (riuscita, dado1, dado2, soglia)

    `riuscita` è True se la somma dei dadi <= soglia.

    Pre-condizione: deve_tirare_sdadata(stato, colore) è True.
    """
    soglia = soglia_sdadata(colore, stato.round_corrente)
    assert soglia is not None, (
        f"tira_sdadata chiamata fuori round valido: "
        f"colore={colore}, round={stato.round_corrente}"
    )

    dado1 = rng.randint(1, 6)
    dado2 = rng.randint(1, 6)
    somma = dado1 + dado2

    riuscita = somma <= soglia
    return (riuscita, dado1, dado2, soglia)


# ─────────────────────────────────────────────────────────────────────────
#  TERMINAZIONE PARTITA
# ─────────────────────────────────────────────────────────────────────────

def termina_partita_per_sdadata(stato: StatoPartita, colore_che_sdada: str) -> None:
    """
    Marca la partita come terminata per sdadata riuscita.
    Determina il vincitore via cascata di criteri (specifica 7.3).

    Non gestisce direttamente i giocatori successivi del round corrente:
    il loro turno semplicemente non avverrà perché stato.terminata=True.
    """
    stato.terminata = True
    stato.motivo_fine = "sdadata"
    stato.vincitore = determina_vincitore(stato)


def termina_partita_per_cap_sicurezza(stato: StatoPartita) -> None:
    """
    Termina la partita per cap di sicurezza (specifica 7.2.6).
    Si applica al termine del round 60.
    """
    stato.terminata = True
    stato.motivo_fine = "cap_sicurezza"
    stato.vincitore = determina_vincitore(stato)


def deve_attivare_cap_sicurezza(stato: StatoPartita, ultimo_giocatore_del_round: bool) -> bool:
    """
    Verifica se è il momento di attivare il cap di sicurezza.
    Si attiva alla fine del round ROUND_CAP_SICUREZZA (60), cioè dopo che
    l'ultimo giocatore vivo di quel round ha completato il suo turno.
    """
    if not ultimo_giocatore_del_round:
        return False
    return stato.round_corrente >= ROUND_CAP_SICUREZZA


# ─────────────────────────────────────────────────────────────────────────
#  HELPER INTEGRATIVO: gestione fine turno
# ─────────────────────────────────────────────────────────────────────────

def gestisci_fine_turno(
    stato: StatoPartita,
    colore: str,
    rng: random.Random,
) -> dict:
    """
    Helper integrativo da chiamare al TERMINE di un turno (DOPO la fase 4
    di pesca carta, ma PRIMA di chiamare avanza_turno).

    Esegue, nell'ordine:
    1. Verifica vittoria immediata per obiettivo (in caso fosse già scattata
       durante un attacco, qui non fa nulla).
    2. Se la partita NON è ancora terminata: verifica se deve tirare sdadata.
       Se sì, tira e se riesce termina la partita.
    3. Se la partita NON è ancora terminata: verifica cap di sicurezza
       (round 60 + ultimo giocatore vivo).

    Restituisce un dict con info di logging:
        {
            "sdadata_tirata": bool,
            "sdadata_dadi": (dado1, dado2) o None,
            "sdadata_soglia": int o None,
            "sdadata_riuscita": bool,
            "cap_sicurezza_attivato": bool,
            "partita_terminata": bool,
            "vincitore": colore o None,
            "motivo_fine": stringa o None,
        }
    """
    info = {
        "sdadata_tirata": False,
        "sdadata_dadi": None,
        "sdadata_soglia": None,
        "sdadata_riuscita": False,
        "cap_sicurezza_attivato": False,
        "partita_terminata": stato.terminata,
        "vincitore": stato.vincitore,
        "motivo_fine": stato.motivo_fine,
    }

    # 1. Se la partita è già terminata (es. obiettivo completato durante attacco)
    if stato.terminata:
        return info

    # 2. Sdadata obbligatoria
    if deve_tirare_sdadata(stato, colore):
        riuscita, d1, d2, soglia = tira_sdadata(stato, colore, rng)
        info["sdadata_tirata"] = True
        info["sdadata_dadi"] = (d1, d2)
        info["sdadata_soglia"] = soglia
        info["sdadata_riuscita"] = riuscita

        if riuscita:
            termina_partita_per_sdadata(stato, colore)
            info["partita_terminata"] = True
            info["vincitore"] = stato.vincitore
            info["motivo_fine"] = stato.motivo_fine
            return info

    # 3. Cap di sicurezza: è l'ultimo giocatore vivo di questo round?
    vivi = stato.giocatori_vivi()
    ultimo_del_round = (vivi and vivi[-1] == colore)

    if deve_attivare_cap_sicurezza(stato, ultimo_del_round):
        termina_partita_per_cap_sicurezza(stato)
        info["cap_sicurezza_attivato"] = True
        info["partita_terminata"] = True
        info["vincitore"] = stato.vincitore
        info["motivo_fine"] = stato.motivo_fine

    return info
