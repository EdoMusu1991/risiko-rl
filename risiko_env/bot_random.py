"""
bot_random.py — Bot random per gli avversari nell'environment.

Strategia COMPLETAMENTE RANDOM. Per ogni decisione, sceglie azione random
tra quelle legali. Niente euristiche, niente filtri ratio, niente preferenze.

IMPORTANTE: questo bot deve giocare con la stessa "stupidità" del bot RL
quando è random (non addestrato). Altrimenti c'è asimmetria: il bot RL
deve battere avversari smart, ma parte da random — partita persa in partenza.

Estratto da env.py durante il refactoring.
"""

import random

from .data import ADIACENZE
from .stato import StatoPartita
from .motore import (
    seleziona_due_tris_disgiunti,
    calcola_bonus_tris,
    calcola_rinforzi_base,
    calcola_bonus_continenti,
    gioca_tris,
    piazza_rinforzi,
    territori_attaccabili_da,
    esegui_attacco,
    applica_conquista,
    spostamento_legale,
    esegui_spostamento,
    pesca_carta,
)


def gioca_turno_random(
    stato: StatoPartita,
    colore: str,
    rng: random.Random,
) -> None:
    """
    Esegue un intero turno per `colore` con strategia completamente random.

    Strategia uniforme: per ogni decisione, sceglie azione random tra quelle legali.
    Niente euristiche.
    """
    giocatore = stato.giocatori[colore]
    if not stato.territori_di(colore):
        return  # eliminato

    # ── FASE 1: tris e rinforzi ──────────────────────────────────
    # Tris: random se giocarli o no
    tris_da_giocare = seleziona_due_tris_disgiunti(giocatore.carte)
    bonus_tris = 0
    if tris_da_giocare and rng.random() < 0.5:
        bonus_tris = calcola_bonus_tris(stato, colore, tris_da_giocare)
        gioca_tris(stato, colore, tris_da_giocare)

    rinf_base = calcola_rinforzi_base(stato, colore)
    bonus_cont = calcola_bonus_continenti(stato, colore)
    totale_rinforzi = rinf_base + bonus_cont + bonus_tris

    # Cap 130 (come fa il bot RL)
    armate_correnti = stato.num_armate_di(colore)
    spazio = max(0, 130 - armate_correnti)
    totale_rinforzi = min(totale_rinforzi, spazio)

    territori_propri = stato.territori_di(colore)
    if not territori_propri:
        return

    if totale_rinforzi > 0:
        # Distribuzione casuale uniforme sui propri territori
        distribuzione = {}
        for _ in range(totale_rinforzi):
            t = rng.choice(territori_propri)
            distribuzione[t] = distribuzione.get(t, 0) + 1
        piazza_rinforzi(stato, colore, distribuzione)

    # ── FASE 2: attacchi ─────────────────────────────────────────
    # Numero di attacchi random tra 0 e 3
    # Niente filtro rapporto: attacca anche se sfavorevole
    max_attacchi = rng.randint(0, 3)
    n_attacchi = 0
    while n_attacchi < max_attacchi and not stato.terminata:
        propri_con_armate = [t for t in stato.territori_di(colore)
                             if stato.mappa[t].armate >= 2]
        if not propri_con_armate:
            break

        da = rng.choice(propri_con_armate)
        attaccabili = territori_attaccabili_da(stato, da)
        if not attaccabili:
            n_attacchi += 1
            continue

        verso = rng.choice(attaccabili)
        # NIENTE filtro rapporto: attacca a ratio random
        # 1 lancio per attacco (come bot RL), poi decide random se continuare
        esito = esegui_attacco(stato, colore, da, verso, rng,
                              fermati_dopo_lanci=1)

        # 50% di chance di continuare a tirare se non ha conquistato
        while (not esito.conquistato
               and not stato.terminata
               and rng.random() < 0.5
               and stato.mappa[da].proprietario == colore
               and stato.mappa[da].armate >= 2
               and stato.mappa[verso].proprietario != colore):
            from .motore import attacco_legale
            if not attacco_legale(stato, colore, da, verso):
                break
            esito = esegui_attacco(stato, colore, da, verso, rng,
                                  fermati_dopo_lanci=1)

        if esito.conquistato:
            minimo = esito.num_dadi_ultimo_lancio
            massimo = stato.mappa[da].armate - 1
            if minimo <= massimo:
                # Quantità random tra minimo e massimo
                quantita = rng.randint(minimo, massimo)
                fine = applica_conquista(stato, colore, da, verso,
                                         quantita, esito, rng)
                if fine:
                    return
        n_attacchi += 1
        if stato.terminata:
            return

    # ── FASE 3: spostamento (random, 30%) ────────────────────────
    if rng.random() < 0.3:
        territori_propri = stato.territori_di(colore)
        candidati = []
        for da in territori_propri:
            if stato.mappa[da].armate < 3:
                continue
            for verso in ADIACENZE[da]:
                if stato.mappa[verso].proprietario == colore:
                    candidati.append((da, verso))
        if candidati:
            da, verso = rng.choice(candidati)
            from .motore import _minimo_da_lasciare_per_spostamento
            min_da_lasciare = _minimo_da_lasciare_per_spostamento(stato, da, colore)
            massimo = stato.mappa[da].armate - min_da_lasciare
            if massimo >= 1:
                quantita = rng.randint(1, massimo)
                if spostamento_legale(stato, colore, da, verso, quantita):
                    esegui_spostamento(stato, colore, da, verso, quantita)

    # ── FASE 4: pesca carta ──────────────────────────────────────
    pesca_carta(stato, colore, rng)
