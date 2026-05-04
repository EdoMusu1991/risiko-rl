"""
bot_euristico.py — Bot con regole euristiche minimali.

Progettato come ROLLOUT POLICY per MCTS (Mese 1 Settimana 2).
NON e' un bot intelligente. E' "leggermente meno stupido del random".
Tre regole semplici (specifica ChatGPT):

1. Attacca solo se win prob > SOGLIA_WIN_PROB (default 0.55)
2. Leggera preferenza (60%) per rinforzo confini
3. Lascia almeno MIN_ARMATE_DIETRO=2 armate dietro quando attacca

Tutto il resto resta uguale a bot_random.py:
- Tris al 50% (non sempre)
- Spostamento al 30% (non sempre verso confini)
- Quantita' rinforzi/spostamenti random
- 0-3 attacchi per turno (non aumentato)
- 50% chance continuare lanci

NOTE SUL TUNING (lezione di benchmark):
- Versione iniziale "bot bravo" (tris sempre, spostamento sempre verso
  confine, fino a 5 attacchi, lanci continui se win prob alta) batteva
  random 100/100. TROPPO. ChatGPT aveva ragione: "non deve essere
  intelligente". Una rollout troppo brava rende MCTS cieco perche'
  vince comunque tutto e non distingue azioni promettenti.
- Versione finale: 3 regole atomiche, tutto il resto random. Atteso
  60-75% win rate vs random — buon compromesso.

Riferimento storico: AlphaGo aveva una rollout policy semplicissima
basata su pattern matching, intenzionalmente non strategica.
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
    attacco_legale,
    _minimo_da_lasciare_per_spostamento,
)


# ─────────────────────────────────────────────────────────────────────
#  COSTANTI EURISTICHE
# ─────────────────────────────────────────────────────────────────────

# Soglia di probabilita' di vittoria per decidere se attaccare (regola 1).
# Valori tra 0.50 e 0.65 sono ragionevoli. Default 0.55 = leggermente sopra il
# pari per evitare attacchi inutili senza essere troppo cauto.
SOGLIA_WIN_PROB = 0.55

# Quante armate minime lasciare dietro quando attacchi (regola 3).
# 1 armata e' il minimo regolamentare. Tenerne 2 evita di lasciare territori
# vulnerabili a contrattacchi facili.
MIN_ARMATE_DIETRO = 2


# ─────────────────────────────────────────────────────────────────────
#  TABELLA DI PROBABILITA' DI VITTORIA
# ─────────────────────────────────────────────────────────────────────
#
# Stimata via simulazione 5000 iterazioni per cella. Vedi script di calcolo
# nella sezione "Test" del modulo.
#
# Formato: WIN_PROB_TABLE[(armate_attacco, armate_difesa)] = probabilita
# Coperti i casi rilevanti per partite normali. Per casi non in tabella,
# stima approssimata via win_probability().

_WIN_PROB_TABLE = {
    (2, 1): 0.76, (3, 1): 0.92, (4, 1): 0.97, (5, 1): 0.99, (6, 1): 0.99,
    (3, 2): 0.66, (4, 2): 0.80, (5, 2): 0.89, (6, 2): 0.94, (7, 2): 0.97,
    (4, 3): 0.64, (5, 3): 0.77, (6, 3): 0.85, (7, 3): 0.90, (8, 3): 0.94,
    (5, 4): 0.63, (6, 4): 0.74, (7, 4): 0.82, (8, 4): 0.87, (9, 4): 0.91,
    (6, 5): 0.62, (7, 5): 0.71, (8, 5): 0.82, (9, 5): 0.86, (10, 5): 0.92,
    (7, 6): 0.62, (8, 6): 0.70, (9, 6): 0.78, (10, 6): 0.84,
    (8, 7): 0.62, (9, 7): 0.69, (10, 7): 0.76,
    (9, 8): 0.62, (10, 8): 0.73,
    (10, 9): 0.62,
}


def win_probability(armate_attacco: int, armate_difesa: int) -> float:
    """
    Stima la probabilita' di conquistare un territorio.

    armate_attacco: numero di armate disponibili PER L'ATTACCO (non quelle
                    sul territorio, ma quelle - 1 per quella che resta).
    armate_difesa: armate sul territorio difensore.

    Per coppie non in tabella, fa una interpolazione lineare ragionevole.
    """
    if armate_attacco <= 0 or armate_difesa <= 0:
        return 0.0

    # Lookup diretto
    if (armate_attacco, armate_difesa) in _WIN_PROB_TABLE:
        return _WIN_PROB_TABLE[(armate_attacco, armate_difesa)]

    # Cap: se attacco molto > difesa, ~stima alta
    if armate_attacco >= armate_difesa * 3:
        return 0.95
    # Floor: se difesa molto > attacco, stima bassa
    if armate_difesa >= armate_attacco * 2:
        return 0.20

    # Stima via ratio
    ratio = armate_attacco / armate_difesa
    if ratio >= 2.0:
        return 0.85
    elif ratio >= 1.5:
        return 0.75
    elif ratio >= 1.2:
        return 0.65
    elif ratio >= 1.0:
        return 0.55
    elif ratio >= 0.8:
        return 0.40
    else:
        return 0.25


# ─────────────────────────────────────────────────────────────────────
#  TURN LOGIC
# ─────────────────────────────────────────────────────────────────────

def gioca_turno_euristico(
    stato: StatoPartita,
    colore: str,
    rng: random.Random,
) -> None:
    """
    Esegue un turno con strategia euristica minimale.

    NB: progettato come rollout policy per MCTS, non come bot strategico.
    """
    giocatore = stato.giocatori[colore]
    if not stato.territori_di(colore):
        return

    # ── FASE 1: tris e rinforzi ─────────────────────────────────────
    # Tris: 50% di chance se disponibile (come random, NON sempre)
    tris_da_giocare = seleziona_due_tris_disgiunti(giocatore.carte)
    bonus_tris = 0
    if tris_da_giocare and rng.random() < 0.5:
        bonus_tris = calcola_bonus_tris(stato, colore, tris_da_giocare)
        gioca_tris(stato, colore, tris_da_giocare)

    rinf_base = calcola_rinforzi_base(stato, colore)
    bonus_cont = calcola_bonus_continenti(stato, colore)
    totale_rinforzi = rinf_base + bonus_cont + bonus_tris

    armate_correnti = stato.num_armate_di(colore)
    spazio = max(0, 130 - armate_correnti)
    totale_rinforzi = min(totale_rinforzi, spazio)

    territori_propri = stato.territori_di(colore)
    if not territori_propri:
        return

    if totale_rinforzi > 0:
        # === REGOLA 2 (light): leggera preferenza per territori di confine ===
        # Identifica territori con almeno un vicino nemico
        confini = [
            t for t in territori_propri
            if any(stato.mappa[v].proprietario != colore for v in ADIACENZE[t])
        ]
        # 60% sui confini se ce ne sono, 40% uniforme. Meno aggressivo del 75%.
        distribuzione = {}
        for _ in range(totale_rinforzi):
            if confini and rng.random() < 0.60:
                t = rng.choice(confini)
            else:
                t = rng.choice(territori_propri)
            distribuzione[t] = distribuzione.get(t, 0) + 1
        piazza_rinforzi(stato, colore, distribuzione)

    # ── FASE 2: attacchi (con regole 1 e 3) ──────────────────────────
    # Numero di tentativi: 0-3 (come random, non aumento)
    max_attacchi = rng.randint(0, 3)
    n_attacchi_eseguiti = 0
    n_tentativi = 0
    MAX_TENTATIVI = 8

    while n_attacchi_eseguiti < max_attacchi and n_tentativi < MAX_TENTATIVI:
        n_tentativi += 1
        if stato.terminata:
            return

        # === REGOLA 3: solo territori con armate >= MIN_ARMATE_DIETRO + 1 ===
        propri_attaccanti = [
            t for t in stato.territori_di(colore)
            if stato.mappa[t].armate >= MIN_ARMATE_DIETRO + 1
        ]
        if not propri_attaccanti:
            break

        da = rng.choice(propri_attaccanti)
        attaccabili = territori_attaccabili_da(stato, da)
        if not attaccabili:
            continue

        # === REGOLA 1: scegli target con win_prob >= SOGLIA ===
        target_candidati = []
        armate_attacco = stato.mappa[da].armate - MIN_ARMATE_DIETRO
        for verso in attaccabili:
            armate_dif = stato.mappa[verso].armate
            if win_probability(armate_attacco, armate_dif) >= SOGLIA_WIN_PROB:
                target_candidati.append(verso)

        if not target_candidati:
            continue

        verso = rng.choice(target_candidati)

        # Attacca
        esito = esegui_attacco(stato, colore, da, verso, rng,
                              fermati_dopo_lanci=1)

        # Continua a tirare 50% (come random) se ancora favorevole
        while (not esito.conquistato
               and not stato.terminata
               and rng.random() < 0.5
               and stato.mappa[da].proprietario == colore
               and stato.mappa[da].armate >= MIN_ARMATE_DIETRO + 1
               and stato.mappa[verso].proprietario != colore):
            if not attacco_legale(stato, colore, da, verso):
                break
            esito = esegui_attacco(stato, colore, da, verso, rng,
                                  fermati_dopo_lanci=1)

        if esito.conquistato:
            minimo = esito.num_dadi_ultimo_lancio
            massimo = stato.mappa[da].armate - 1
            if minimo <= massimo:
                # Quantita' random come random bot, NON quantita_target smart
                quantita = rng.randint(minimo, massimo)
                fine = applica_conquista(stato, colore, da, verso,
                                         quantita, esito, rng)
                if fine:
                    return
        n_attacchi_eseguiti += 1

        if stato.terminata:
            return

    # ── FASE 3: spostamento (come random, 30%) ───────────────────────
    # Niente smart "interno -> confine". Random come bot_random.
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
            min_da_lasciare = _minimo_da_lasciare_per_spostamento(stato, da, colore)
            massimo = stato.mappa[da].armate - min_da_lasciare
            if massimo >= 1:
                quantita = rng.randint(1, massimo)
                if spostamento_legale(stato, colore, da, verso, quantita):
                    esegui_spostamento(stato, colore, da, verso, quantita)

    # ── FASE 4: pesca carta ──────────────────────────────────────────
    pesca_carta(stato, colore, rng)
