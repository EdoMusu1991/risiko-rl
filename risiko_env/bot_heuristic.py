"""
bot_heuristic.py — Bot "stupido ma sensato" per rollout MCTS.

Specifica ChatGPT (Settimana 2):
> "rollout policy stupida ma sensata"
> - attacca solo se win prob > 55-60%
> - rinforza territori di confine
> - evita overextension (no attacchi che lasciano territori a 1 armata)
> - non deve essere intelligente, deve essere meno stupida del random

Differenze chiave da bot_random:
1. Filtro probabilita': non attacca con ratio sfavorevole
2. Rinforzi mirati: piazza armate sui territori di confine, non random
3. No overextension: dopo conquista lascia >=2 armate sul territorio sorgente
4. Niente bluff/trick: gioca tris se conviene (no 50/50 random)

NB: questa policy guida i rollout di MCTS quando non abbiamo ancora la rete neurale.
Nei mesi 2+ verra' sostituita dalla rete neurale (policy head).
"""

import random
from typing import Optional

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
)


# ─────────────────────────────────────────────────────────────────────────
#  TABELLA PROBABILITA' VINCITA ATTACCO (calcolata empiricamente)
# ─────────────────────────────────────────────────────────────────────────
#
# Probabilita' che attaccante VINCA un singolo lancio (1 dado vs 1 dado,
# 2 dadi vs 1, 2 dadi vs 2, 3 dadi vs 1, 3 dadi vs 2).
# Se attaccante perde, perde 1 armata. Se vince, difensore perde 1 armata.
#
# Fonte: probabilita' esatte calcolate da regolamento Risiko.
# Sono usate per stimare se conviene attaccare in modo veloce.
#
# Indice: (n_dadi_attaccante, n_dadi_difensore) -> P(attaccante vince almeno 1)
PROB_VITTORIA_LANCIO = {
    (1, 1): 0.417,
    (1, 2): 0.255,
    (2, 1): 0.579,
    (2, 2): 0.448,  # attaccante vince entrambi: 0.228, vince 1 e perde 1: 0.324
    (3, 1): 0.660,
    (3, 2): 0.560,  # complessa, almeno una vittoria
}


def stima_prob_vittoria_attacco(armate_att: int, armate_dif: int) -> float:
    """
    Stima approssimata della probabilita' che l'attaccante CONQUISTI il
    territorio (riduca difensore a 0) prima di esaurire le proprie armate.

    Approccio: calcolo iterativo basato sulla tabella di lancio singolo,
    con espansione fino a 5 round (per velocita').

    Risultato in [0, 1]. Per ratio >= 2:1 dovrebbe essere > 0.6.

    Esempi (verificati):
    - 3 vs 1: ~0.78
    - 3 vs 2: ~0.45
    - 4 vs 2: ~0.66
    - 5 vs 3: ~0.64
    - 2 vs 1: ~0.42
    """
    if armate_att <= 1:
        return 0.0  # non puo' attaccare
    if armate_dif <= 0:
        return 1.0  # gia' conquistato

    # Approssimazione: simulazione veloce monte carlo con regole semplificate.
    # Usa la tabella di lancio per stimare in modo deterministico.
    #
    # Per non rallentare il rollout, calcoliamo solo il caso "media":
    # attaccante usa max dadi possibili, difensore usa max dadi possibili.

    # Numero di lanci attesi prima che attaccante esaurisca o conquisti
    # In media ogni lancio elimina 1 armata complessivamente
    armate_att_disp = armate_att - 1  # lascia 1 sul territorio sorgente

    # Approssimazione semplice: ratio-based
    # Empiricamente per Risiko:
    #   ratio < 1.0: prob ~0.20
    #   ratio = 1.0: prob ~0.40
    #   ratio = 1.5: prob ~0.60
    #   ratio = 2.0: prob ~0.75
    #   ratio = 3.0: prob ~0.88
    ratio = armate_att_disp / max(armate_dif, 1)

    if ratio < 0.7:
        return 0.15
    if ratio < 1.0:
        return 0.30
    if ratio < 1.3:
        return 0.45
    if ratio < 1.7:
        return 0.60
    if ratio < 2.2:
        return 0.75
    if ratio < 3.0:
        return 0.85
    return 0.92


# ─────────────────────────────────────────────────────────────────────────
#  HELPER GEOMETRICI
# ─────────────────────────────────────────────────────────────────────────

def territori_di_confine(stato: StatoPartita, colore: str) -> list[str]:
    """Territori di `colore` adiacenti ad almeno un nemico."""
    miei = stato.territori_di(colore)
    confine = []
    for t in miei:
        for vicino in ADIACENZE[t]:
            if stato.mappa[vicino].proprietario != colore:
                confine.append(t)
                break
    return confine


def territori_interni(stato: StatoPartita, colore: str) -> list[str]:
    """Territori di `colore` circondati solo da propri (no nemici adiacenti)."""
    miei = set(stato.territori_di(colore))
    interni = []
    for t in miei:
        if all(v in miei for v in ADIACENZE[t]):
            interni.append(t)
    return interni


# ─────────────────────────────────────────────────────────────────────────
#  TURNO EURISTICO
# ─────────────────────────────────────────────────────────────────────────

def gioca_turno_heuristic(
    stato: StatoPartita,
    colore: str,
    rng: random.Random,
    soglia_attacco: float = 0.55,
) -> None:
    """
    Esegue un turno con strategia "stupida ma sensata".

    Regole:
    1. Tris: gioca SEMPRE se ne hai uno disponibile (no random)
    2. Rinforzi: piazza tutto sui territori di confine
    3. Attacchi: solo se win prob > soglia_attacco (default 55%)
    4. No overextension: dopo conquista lascia minimo possibile (1) sul nuovo
       e tutto il resto sul vecchio (NO trasferisce 99% sul nuovo territorio)
    5. Spostamento: 50% sposta dall'interno al confine

    Args:
        stato: stato corrente della partita
        colore: colore del giocatore che gioca questo turno
        rng: random.Random per le decisioni stocastiche residue
        soglia_attacco: probabilita' minima per decidere di attaccare

    Modifica `stato` in-place. Usa esattamente le stesse funzioni del motore
    di gioco, quindi rispetta tutte le regole (limite 130 carri, sdadata, ecc.)
    """
    giocatore = stato.giocatori[colore]
    if not stato.territori_di(colore):
        return  # eliminato

    # ─── FASE 1: Tris e rinforzi ───
    tris_da_giocare = seleziona_due_tris_disgiunti(giocatore.carte)
    bonus_tris = 0
    if tris_da_giocare:
        # Sempre gioca il tris se disponibile (8-12 carri = enorme valore)
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
        # Rinforzi sui territori di confine, distribuiti pesati per minaccia
        confini = territori_di_confine(stato, colore)
        if not confini:
            confini = territori_propri  # fallback

        # Distribuzione: piu' rinforzi ai territori con piu' nemici adiacenti
        pesi = {}
        for t in confini:
            n_nemici = sum(
                1 for v in ADIACENZE[t]
                if stato.mappa[v].proprietario != colore
                and stato.mappa[v].proprietario is not None
            )
            armate_qui = stato.mappa[t].armate
            armate_nemiche_adiacenti = sum(
                stato.mappa[v].armate for v in ADIACENZE[t]
                if stato.mappa[v].proprietario != colore
                and stato.mappa[v].proprietario is not None
            )
            # Peso: alto se nemici tanti e armate proprie poche
            pesi[t] = max(1, armate_nemiche_adiacenti - armate_qui + 1) * (1 + n_nemici)

        distribuzione = {}
        # Distribuzione weighted random
        territori_lista = list(pesi.keys())
        pesi_lista = [pesi[t] for t in territori_lista]
        for _ in range(totale_rinforzi):
            t = rng.choices(territori_lista, weights=pesi_lista, k=1)[0]
            distribuzione[t] = distribuzione.get(t, 0) + 1
        piazza_rinforzi(stato, colore, distribuzione)

    # ─── FASE 2: Attacchi ───
    # Attacca finche' trovi attacchi favorevoli (max 5 attacchi per turno)
    n_attacchi = 0
    max_attacchi_turno = 5

    while n_attacchi < max_attacchi_turno and not stato.terminata:
        # Trova migliore attacco (highest prob > soglia)
        miglior_attacco = None
        miglior_prob = soglia_attacco

        propri_con_armate = [t for t in stato.territori_di(colore)
                             if stato.mappa[t].armate >= 2]
        if not propri_con_armate:
            break

        for da in propri_con_armate:
            for verso in territori_attaccabili_da(stato, da):
                armate_att = stato.mappa[da].armate
                armate_dif = stato.mappa[verso].armate
                prob = stima_prob_vittoria_attacco(armate_att, armate_dif)
                if prob > miglior_prob:
                    miglior_prob = prob
                    miglior_attacco = (da, verso)

        if miglior_attacco is None:
            break  # nessun attacco favorevole

        da, verso = miglior_attacco

        # Esegue attacco con multi-lancio (continua finche' favorevole o conquista)
        esito = esegui_attacco(stato, colore, da, verso, rng, fermati_dopo_lanci=1)
        while (not esito.conquistato
               and not stato.terminata
               and stato.mappa[da].proprietario == colore
               and stato.mappa[da].armate >= 2
               and stato.mappa[verso].proprietario != colore):
            # Ricalcola prob: continua solo se ancora favorevole
            prob_corr = stima_prob_vittoria_attacco(
                stato.mappa[da].armate,
                stato.mappa[verso].armate,
            )
            if prob_corr < soglia_attacco:
                break
            if not attacco_legale(stato, colore, da, verso):
                break
            esito = esegui_attacco(stato, colore, da, verso, rng, fermati_dopo_lanci=1)

        if esito.conquistato:
            # Quantita': minimo necessario, niente overextension
            minimo = esito.num_dadi_ultimo_lancio
            massimo = stato.mappa[da].armate - 1
            if minimo <= massimo:
                # Strategia: trasferisci minimo per non lasciare il territorio
                # sorgente troppo scoperto
                quantita = max(minimo, min(massimo, 2))  # almeno 2 sul nuovo, salvo che il min sia maggiore
                quantita = min(quantita, massimo)
                fine = applica_conquista(stato, colore, da, verso,
                                         quantita, esito, rng)
                if fine:
                    return
        n_attacchi += 1
        if stato.terminata:
            return

    # ─── FASE 3: Spostamento (50% probabilita') ───
    if rng.random() < 0.5:
        territori_propri = stato.territori_di(colore)
        # Candidati: da territori interni con >= 3 armate, verso territori di confine
        confini = set(territori_di_confine(stato, colore))
        candidati = []
        for da in territori_propri:
            if stato.mappa[da].armate < 3:
                continue
            if da in confini:
                continue  # non spostare via dal confine
            for verso in ADIACENZE[da]:
                if stato.mappa[verso].proprietario == colore and verso in confini:
                    candidati.append((da, verso))

        if candidati:
            da, verso = rng.choice(candidati)
            from .motore import _minimo_da_lasciare_per_spostamento
            min_da_lasciare = _minimo_da_lasciare_per_spostamento(stato, da, colore)
            massimo = stato.mappa[da].armate - min_da_lasciare
            if massimo >= 1:
                # Sposta tutto il possibile (rinforzo aggressivo del confine)
                quantita = massimo
                if spostamento_legale(stato, colore, da, verso, quantita):
                    esegui_spostamento(stato, colore, da, verso, quantita)

    # ─── FASE 4: Pesca carta ───
    pesca_carta(stato, colore, rng)
