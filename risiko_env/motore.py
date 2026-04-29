"""
motore.py — Modulo 3: Le 4 fasi del turno.

Espone le primitive di gioco: funzioni che applicano azioni allo stato.
Niente decisioni, niente bot — solo "applica questa azione, è legale?".

Le 4 fasi:
- TRIS_E_RINFORZI: gioca tris, calcola rinforzi, piazza
- ATTACCHI: combattimenti, conquiste, eliminazioni
- SPOSTAMENTO: 1 spostamento finale tra territori adiacenti
- PESCA_CARTA: pesca 1 se conquistato (rispetta limite 7)

Specifica di riferimento: risiko_specifica_v1.2.md sezione 4.
"""

import random
from typing import Optional

from .data import (
    CONTINENTI,
    BONUS_CONTINENTE,
    ADIACENZE,
    SIMBOLO_CARTA,
    FANTE,
    CANNONE,
    CAVALLO,
    JOLLY,
    BONUS_TRIS_3_UGUALI,
    BONUS_TRIS_3_DIVERSI,
    BONUS_TRIS_JOLLY_PIU_2,
    BONUS_TERRITORIO_IN_CARTA,
    MAX_ARMATE_TOTALI,
    MAX_CARTE_MANO,
)
from .stato import (
    StatoPartita,
    Carta,
    Giocatore,
)
from .combattimento import (
    num_dadi_attaccante,
    risolvi_lancio,
)
from .obiettivi import ha_completato_obiettivo


# ═════════════════════════════════════════════════════════════════════════
#  FASE 1 — TRIS_E_RINFORZI
# ═════════════════════════════════════════════════════════════════════════

def calcola_rinforzi_base(stato: StatoPartita, colore: str) -> int:
    """
    Rinforzi base: floor(territori / 3). Nessun minimo (specifica 4.1.1).
    """
    n = stato.num_territori_di(colore)
    return n // 3


def calcola_bonus_continenti(stato: StatoPartita, colore: str) -> int:
    """
    Somma dei bonus per ogni continente posseduto interamente
    (specifica 4.1.2).
    """
    territori_propri = set(stato.territori_di(colore))
    bonus = 0
    for cont, terrs in CONTINENTI.items():
        if all(t in territori_propri for t in terrs):
            bonus += BONUS_CONTINENTE[cont]
    return bonus


# ─────────────────────────────────────────────────────────────────────────
#  TRIS
# ─────────────────────────────────────────────────────────────────────────

def trova_tutti_i_tris(carte: list[Carta]) -> list[tuple[list[Carta], int]]:
    """
    Trova TUTTE le combinazioni valide di tris dalle carte in mano.
    Restituisce lista di (3 carte, bonus_armate) senza ancora applicare il bonus territori.

    Tipi di tris (specifica 4.1.3):
    - 3 carte uguali (3 fanti, 3 cannoni, 3 cavalli) → +8
    - 3 carte diverse (1 fante + 1 cannone + 1 cavallo) → +10
    - 1 jolly + 2 carte uguali → +12

    NB: questa funzione genera tutti i tris POSSIBILI (anche sovrapposti).
    Sarà compito di chi gioca i tris assicurarsi di non riusare la stessa carta.
    Per il bot RL useremo un metodo più semplice: cerca al massimo 2 tris disgiunti.
    """
    tris_trovati: list[tuple[list[Carta], int]] = []

    # Separa per simbolo
    fanti = [c for c in carte if c.simbolo == FANTE]
    cannoni = [c for c in carte if c.simbolo == CANNONE]
    cavalli = [c for c in carte if c.simbolo == CAVALLO]
    jolly = [c for c in carte if c.is_jolly]

    # Tris 3 uguali
    for gruppo in (fanti, cannoni, cavalli):
        if len(gruppo) >= 3:
            tris_trovati.append((gruppo[:3], BONUS_TRIS_3_UGUALI))

    # Tris 3 diversi (1 fante + 1 cannone + 1 cavallo)
    if fanti and cannoni and cavalli:
        tris_trovati.append((
            [fanti[0], cannoni[0], cavalli[0]],
            BONUS_TRIS_3_DIVERSI,
        ))

    # Tris jolly + 2 uguali
    for j in jolly:
        for gruppo in (fanti, cannoni, cavalli):
            if len(gruppo) >= 2:
                tris_trovati.append((
                    [j, gruppo[0], gruppo[1]],
                    BONUS_TRIS_JOLLY_PIU_2,
                ))

    return tris_trovati


def seleziona_due_tris_disgiunti(carte: list[Carta]) -> list[tuple[list[Carta], int]]:
    """
    Seleziona fino a 2 tris disgiunti (senza carte in comune) di valore
    massimo combinato. Strategia greedy: prende il tris di valore più alto,
    rimuove le sue carte, cerca un secondo tris.

    Restituisce 0, 1 o 2 tris (lista di tuple (carte_del_tris, bonus_armate)).
    """
    tris_disponibili = trova_tutti_i_tris(carte)
    if not tris_disponibili:
        return []

    # Ordina per bonus decrescente per prendere prima il più valido
    tris_disponibili.sort(key=lambda x: x[1], reverse=True)

    # Primo tris: il più valido
    primo_tris, primo_bonus = tris_disponibili[0]
    risultato = [(primo_tris, primo_bonus)]

    # Secondo tris: cerca tra le carte rimanenti (escluse quelle del primo tris)
    carte_rimaste = [c for c in carte if c not in primo_tris]
    secondi = trova_tutti_i_tris(carte_rimaste)
    if secondi:
        secondi.sort(key=lambda x: x[1], reverse=True)
        risultato.append(secondi[0])

    return risultato


def calcola_bonus_tris(
    stato: StatoPartita,
    colore: str,
    tris_giocati: list[tuple[list[Carta], int]],
) -> int:
    """
    Calcola il bonus armate totale dei tris giocati, incluso il bonus
    territori (+2 per ogni carta del tris di territorio posseduto, specifica 4.1.3).
    """
    territori_propri = set(stato.territori_di(colore))
    totale = 0

    for carte_tris, bonus_base in tris_giocati:
        totale += bonus_base
        # Bonus territori per ogni carta del tris (jolly esclusi)
        for c in carte_tris:
            if not c.is_jolly and c.territorio in territori_propri:
                totale += BONUS_TERRITORIO_IN_CARTA

    return totale


def gioca_tris(
    stato: StatoPartita,
    colore: str,
    tris_da_giocare: list[tuple[list[Carta], int]],
) -> None:
    """
    Rimuove le carte dei tris giocati dalla mano del giocatore e le mette
    nella pila scarti. Non distribuisce le armate (lo fa piazza_rinforzi).
    """
    giocatore = stato.giocatori[colore]
    carte_da_rimuovere: list[Carta] = []
    for carte_tris, _ in tris_da_giocare:
        carte_da_rimuovere.extend(carte_tris)

    # Rimuovi dalla mano (usiamo identità di oggetto, non valore)
    for c in carte_da_rimuovere:
        giocatore.carte.remove(c)

    # Sposta in scarti
    stato.pila_scarti.extend(carte_da_rimuovere)


# ─────────────────────────────────────────────────────────────────────────
#  PIAZZAMENTO RINFORZI
# ─────────────────────────────────────────────────────────────────────────

def piazza_rinforzi(
    stato: StatoPartita,
    colore: str,
    distribuzione: dict[str, int],
) -> int:
    """
    Piazza armate sui territori indicati.

    `distribuzione`: dict territorio → numero_armate da aggiungere
    Ogni territorio in distribuzione deve essere posseduto dal giocatore.

    Applica il cap 130 (specifica 4.1.5): se sommando si superebbe
    il cap, le armate eccedenti vengono perse.

    Restituisce il numero di armate effettivamente piazzate (utile per logging).
    """
    armate_attuali = stato.num_armate_di(colore)
    spazio_disponibile = max(0, MAX_ARMATE_TOTALI - armate_attuali)
    da_piazzare = sum(distribuzione.values())

    if da_piazzare <= spazio_disponibile:
        # Tutto entra
        for t, n in distribuzione.items():
            assert stato.mappa[t].proprietario == colore, (
                f"Tentativo di piazzare su territorio non proprio: {t}"
            )
            stato.mappa[t].armate += n
        return da_piazzare

    # Eccedenza: distribuisci proporzionalmente fino al cap
    # (alternativa: piazza in ordine fino a esaurimento)
    # Strategia semplice: piazza in ordine, ferma quando saturi
    rimanenti = spazio_disponibile
    piazzate = 0
    for t, n in distribuzione.items():
        assert stato.mappa[t].proprietario == colore
        if rimanenti <= 0:
            break
        prendi = min(n, rimanenti)
        stato.mappa[t].armate += prendi
        rimanenti -= prendi
        piazzate += prendi

    return piazzate


# ═════════════════════════════════════════════════════════════════════════
#  FASE 2 — ATTACCHI
# ═════════════════════════════════════════════════════════════════════════

def attacco_legale(stato: StatoPartita, colore_attaccante: str,
                   da: str, verso: str) -> bool:
    """
    Verifica se l'attacco da `da` verso `verso` è legale.

    Condizioni (specifica 4.2):
    - `da` deve essere posseduto da `colore_attaccante`
    - `verso` deve essere posseduto da un AVVERSARIO (non vuoto, non proprio)
    - `da` e `verso` devono essere adiacenti
    - `da` deve avere almeno 2 armate
    """
    if da not in stato.mappa or verso not in stato.mappa:
        return False
    sd = stato.mappa[da]
    sv = stato.mappa[verso]

    if sd.proprietario != colore_attaccante:
        return False
    if sv.proprietario is None or sv.proprietario == colore_attaccante:
        return False
    if verso not in ADIACENZE[da]:
        return False
    if sd.armate < 2:
        return False

    return True


def territori_attaccabili_da(stato: StatoPartita, da: str) -> list[str]:
    """
    Lista dei territori attaccabili da `da`:
    - adiacenti
    - di un avversario (non propri, non vuoti)
    Pre-condizione: `da` deve avere armate >= 2.
    """
    sd = stato.mappa.get(da)
    if not sd or sd.armate < 2 or sd.proprietario is None:
        return []
    colore = sd.proprietario
    return [
        v for v in ADIACENZE[da]
        if stato.mappa[v].proprietario not in (None, colore)
    ]


# ─────────────────────────────────────────────────────────────────────────
#  ESITO DI UN ATTACCO COMPLETO (multi-lancio fino a fine)
# ─────────────────────────────────────────────────────────────────────────

class EsitoAttacco:
    """
    Risultato di un attacco completo (uno o più lanci consecutivi sullo
    stesso territorio difensore, fino a fermarsi/conquistare/troppo deboli).

    Attributi:
        conquistato: True se il difensore è stato ridotto a 0 armate
        armate_attaccante_finali: armate rimaste sul territorio attaccante
            (PRIMA dello spostamento di conquista)
        armate_difensore_finali: armate rimaste sul difensore
            (0 se conquistato)
        perdite_totali_attaccante: armate perse dall'attaccante
        perdite_totali_difensore: armate perse dal difensore
        num_dadi_ultimo_lancio: dadi dell'attaccante nell'ultimo lancio
            (usato per determinare il minimo da spostare in caso di conquista)
        num_lanci: quanti lanci di dadi sono stati fatti
    """
    def __init__(self):
        self.conquistato = False
        self.armate_attaccante_finali = 0
        self.armate_difensore_finali = 0
        self.perdite_totali_attaccante = 0
        self.perdite_totali_difensore = 0
        self.num_dadi_ultimo_lancio = 0
        self.num_lanci = 0


def esegui_attacco(
    stato: StatoPartita,
    colore_attaccante: str,
    da: str,
    verso: str,
    rng: random.Random,
    fermati_dopo_lanci: Optional[int] = None,
) -> EsitoAttacco:
    """
    Esegue un attacco da `da` a `verso`. Tira dadi finché:
    - il difensore va a 0 (conquista) → si ferma
    - l'attaccante non può più tirare (armate <= 1) → si ferma
    - opzionalmente, dopo `fermati_dopo_lanci` lanci

    NON applica lo spostamento di conquista — quello va fatto separatamente
    chiamando applica_conquista() se esito.conquistato.

    Specifica 4.2.4 e 4.2.5.
    """
    assert attacco_legale(stato, colore_attaccante, da, verso), (
        f"Attacco illegale: {colore_attaccante} da {da} a {verso}"
    )

    sd = stato.mappa[da]
    sv = stato.mappa[verso]

    esito = EsitoAttacco()

    while sd.armate >= 2 and sv.armate >= 1:
        # Controllo limite lanci
        if fermati_dopo_lanci is not None and esito.num_lanci >= fermati_dopo_lanci:
            break

        perdite_att, perdite_dif, _, _ = risolvi_lancio(sd.armate, sv.armate, rng)
        sd.armate -= perdite_att
        sv.armate -= perdite_dif

        esito.perdite_totali_attaccante += perdite_att
        esito.perdite_totali_difensore += perdite_dif
        esito.num_dadi_ultimo_lancio = num_dadi_attaccante(sd.armate + perdite_att)
        esito.num_lanci += 1

        # Sicurezza: se non si fa progresso (entrambi 0 perdite), exit
        if perdite_att == 0 and perdite_dif == 0:
            break

    esito.conquistato = (sv.armate == 0)
    esito.armate_attaccante_finali = sd.armate
    esito.armate_difensore_finali = sv.armate

    return esito


# ─────────────────────────────────────────────────────────────────────────
#  CONQUISTA: spostamento armate, eliminazione giocatore, ruba carte
# ─────────────────────────────────────────────────────────────────────────

def applica_conquista(
    stato: StatoPartita,
    colore_attaccante: str,
    da: str,
    verso: str,
    armate_da_spostare: int,
    esito: EsitoAttacco,
    rng: random.Random,
) -> bool:
    """
    Applica la conquista del territorio `verso` (specifica 4.2.6 - 4.2.8).

    Pre-condizioni:
    - esito.conquistato è True (verso ha 0 armate)
    - `armate_da_spostare` è in [num_dadi_ultimo_lancio, armate_da-1]

    Operazioni:
    1. Cambia proprietario di `verso` a colore_attaccante
    2. Sposta armate_da_spostare da `da` a `verso`
    3. Incrementa contatore conquiste del giocatore
    4. Se il difensore originale è stato eliminato (0 territori), ruba le sue carte
    5. Verifica se l'attaccante ha completato il suo obiettivo (vittoria immediata)

    Restituisce True se la partita è terminata per vittoria immediata.
    """
    assert esito.conquistato
    sd = stato.mappa[da]
    sv = stato.mappa[verso]

    # Validazione armate da spostare (specifica 4.2.6)
    minimo = esito.num_dadi_ultimo_lancio
    massimo = sd.armate - 1
    assert minimo <= armate_da_spostare <= massimo, (
        f"Armate da spostare {armate_da_spostare} fuori range [{minimo}, {massimo}]"
    )

    # Identifica il difensore (per controllo eliminazione)
    colore_difensore = sv.proprietario  # ATTENZIONE: già cambiato a None? No, è solo armate=0
    # In realtà sv.proprietario è ancora il colore del difensore qui
    assert colore_difensore is not None and colore_difensore != colore_attaccante

    # Esegui conquista
    sd.armate -= armate_da_spostare
    sv.proprietario = colore_attaccante
    sv.armate = armate_da_spostare

    # Aggiorna contatore conquiste
    stato.conquiste_turno_corrente[colore_attaccante] = (
        stato.conquiste_turno_corrente.get(colore_attaccante, 0) + 1
    )

    # ── Verifica eliminazione del difensore ───────────────────────
    difensore_eliminato = (stato.num_territori_di(colore_difensore) == 0)
    if difensore_eliminato:
        _elimina_giocatore(stato, colore_difensore, colore_attaccante, rng)

    # ── Verifica vittoria immediata per obiettivo ────────────────
    if ha_completato_obiettivo(stato, colore_attaccante):
        stato.terminata = True
        stato.vincitore = colore_attaccante
        stato.motivo_fine = "obiettivo_completato"
        return True

    return False


def _elimina_giocatore(
    stato: StatoPartita,
    colore_eliminato: str,
    colore_uccisore: str,
    rng: random.Random,
) -> None:
    """
    Elimina il giocatore: trasferisce le sue carte all'uccisore (specifica 4.2.8).
    Se il totale supera 7, le eccedenti vanno in pila scarti (scelte a caso).
    """
    eliminato = stato.giocatori[colore_eliminato]
    uccisore = stato.giocatori[colore_uccisore]

    eliminato.vivo = False

    # Trasferisci carte
    carte_da_trasferire = list(eliminato.carte)
    eliminato.carte.clear()

    # Quante carte stiamo aggiungendo all'uccisore?
    spazio = MAX_CARTE_MANO - len(uccisore.carte)

    if len(carte_da_trasferire) <= spazio:
        # Tutte entrano
        uccisore.carte.extend(carte_da_trasferire)
    else:
        # Alcune entrano, le altre vanno scartate (scelte a caso)
        rng.shuffle(carte_da_trasferire)
        uccisore.carte.extend(carte_da_trasferire[:spazio])
        stato.pila_scarti.extend(carte_da_trasferire[spazio:])


# ═════════════════════════════════════════════════════════════════════════
#  FASE 3 — SPOSTAMENTO
# ═════════════════════════════════════════════════════════════════════════

def spostamento_legale(stato: StatoPartita, colore: str,
                       da: str, verso: str, quantita: int) -> bool:
    """
    Verifica se uno spostamento è legale (specifica 4.3).

    Condizioni:
    - `da` e `verso` posseduti dal giocatore
    - adiacenti
    - quantità nel range valido (vedi minimo_da_lasciare)
    """
    if da not in stato.mappa or verso not in stato.mappa:
        return False
    sd = stato.mappa[da]
    sv = stato.mappa[verso]

    if sd.proprietario != colore or sv.proprietario != colore:
        return False
    if verso not in ADIACENZE[da]:
        return False

    minimo_da_lasciare = _minimo_da_lasciare_per_spostamento(stato, da, colore)
    massimo_spostabile = sd.armate - minimo_da_lasciare

    if quantita < 1 or quantita > massimo_spostabile:
        return False
    return True


def _minimo_da_lasciare_per_spostamento(stato: StatoPartita, territorio: str,
                                        colore: str) -> int:
    """
    Specifica 4.3.2:
    - 1 armata se TUTTI i confinanti sono propri (territorio in nicchia)
    - 2 armate altrimenti
    """
    confinanti = ADIACENZE[territorio]
    tutti_propri = all(
        stato.mappa[c].proprietario == colore for c in confinanti
    )
    return 1 if tutti_propri else 2


def esegui_spostamento(stato: StatoPartita, colore: str,
                       da: str, verso: str, quantita: int) -> None:
    """
    Esegue lo spostamento finale (specifica 4.3).
    Pre-condizione: spostamento_legale(...) deve essere True.
    """
    assert spostamento_legale(stato, colore, da, verso, quantita)
    stato.mappa[da].armate -= quantita
    stato.mappa[verso].armate += quantita


# ═════════════════════════════════════════════════════════════════════════
#  FASE 4 — PESCA CARTA
# ═════════════════════════════════════════════════════════════════════════

def pesca_carta(stato: StatoPartita, colore: str, rng: random.Random) -> bool:
    """
    Pesca 1 carta dal mazzo se il giocatore ha conquistato almeno 1 territorio
    in questo turno e non ha già 7 carte (specifica 4.4).

    Restituisce True se ha pescato.
    """
    giocatore = stato.giocatori[colore]
    if stato.conquiste_turno_corrente.get(colore, 0) == 0:
        return False  # Niente conquiste, niente pesca
    if not giocatore.puo_pescare():
        return False  # Già a 7 carte

    # Reshuffling se mazzo vuoto (specifica 4.4 e 5.3)
    if not stato.mazzo_attivo:
        if not stato.pila_scarti:
            # Caso impossibile: mazzo + scarti vuoti significa che tutte le 44 carte
            # sono in mano, ma con max 7 per giocatore × 4 = 28, non si arriva mai
            return False
        stato.mazzo_attivo = list(stato.pila_scarti)
        rng.shuffle(stato.mazzo_attivo)
        stato.pila_scarti = []

    carta = stato.mazzo_attivo.pop()
    giocatore.carte.append(carta)
    return True


# ═════════════════════════════════════════════════════════════════════════
#  AVANZAMENTO TURNO E ROUND
# ═════════════════════════════════════════════════════════════════════════

def prossimo_giocatore(stato: StatoPartita) -> Optional[str]:
    """
    Restituisce il colore del prossimo giocatore vivo nell'ordine di mano,
    oppure None se la partita è terminata.

    Se il giocatore corrente è l'ultimo del round, passa al round successivo.
    """
    if stato.terminata:
        return None

    vivi = stato.giocatori_vivi()
    if not vivi:
        return None

    if stato.giocatore_corrente is None:
        return vivi[0]

    try:
        idx = vivi.index(stato.giocatore_corrente)
    except ValueError:
        # Il giocatore corrente non è più vivo (è stato eliminato durante il turno)
        # Trova il prossimo nell'ordine canonico
        from .data import COLORI_GIOCATORI
        idx_canonico = COLORI_GIOCATORI.index(stato.giocatore_corrente)
        # Trova il prossimo vivo in ordine canonico
        for i in range(1, 5):
            cand = COLORI_GIOCATORI[(idx_canonico + i) % 4]
            if cand in vivi:
                return cand
        return None

    if idx + 1 < len(vivi):
        # Ancora qualcuno nel round
        return vivi[idx + 1]
    else:
        # Fine del round, si ricomincia dal primo
        return vivi[0]


def avanza_turno(stato: StatoPartita) -> None:
    """
    Avanza al prossimo turno:
    - Se eravamo l'ultimo giocatore vivo del round, incrementa il round
    - Imposta il giocatore corrente
    - Resetta il contatore conquiste del nuovo giocatore corrente
    """
    if stato.terminata:
        return

    vivi = stato.giocatori_vivi()
    if not vivi:
        return

    nuovo = prossimo_giocatore(stato)
    if nuovo is None:
        return

    # Se il nuovo è il primo della lista vivi (siamo all'inizio di un round)
    if nuovo == vivi[0] and stato.giocatore_corrente is not None:
        # Verifica che il vecchio fosse l'ultimo vivo del round corrente
        # (se era diverso, semplicemente è cambiato il round)
        if stato.giocatore_corrente == vivi[-1] or (
            stato.giocatore_corrente not in vivi  # eliminato
        ):
            stato.round_corrente += 1

    stato.giocatore_corrente = nuovo
    # Reset conquiste del nuovo giocatore
    stato.conquiste_turno_corrente[nuovo] = 0
