"""
obiettivi.py — Verifica vittoria per obiettivo e calcolo punteggi finali.

Contiene:
- ha_completato_obiettivo: verifica se un giocatore possiede TUTTI i territori
  del proprio obiettivo (vittoria immediata, specifica 7.1.A)
- calcola_punti_in_obiettivo: punti del giocatore dai suoi territori in obiettivo
- calcola_punti_fuori_obiettivo: punti dai territori NON in obiettivo
- determina_vincitore: applica i 3 criteri in cascata della specifica 7.3

Specifica di riferimento: risiko_specifica_v1.2.md sezioni 6.2, 6.3, 7.1.A, 7.3.
"""

from .data import OBIETTIVI, COLORI_GIOCATORI, punti_territorio
from .stato import StatoPartita


# ─────────────────────────────────────────────────────────────────────────
#  VITTORIA IMMEDIATA: OBIETTIVO COMPLETATO
# ─────────────────────────────────────────────────────────────────────────

def ha_completato_obiettivo(stato: StatoPartita, colore: str) -> bool:
    """
    Restituisce True se il giocatore `colore` possiede TUTTI i territori
    del suo obiettivo segreto (vittoria immediata, specifica 7.1.A).

    Se il giocatore non ha un obiettivo assegnato (caso teorico), False.
    """
    giocatore = stato.giocatori[colore]
    if giocatore.obiettivo_id is None:
        return False

    obiettivo = OBIETTIVI[giocatore.obiettivo_id]
    territori_target = obiettivo["territori"]

    # Verifica che tutti i territori target siano del giocatore
    for t in territori_target:
        if stato.mappa[t].proprietario != colore:
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────
#  CALCOLO PUNTEGGI
# ─────────────────────────────────────────────────────────────────────────

def calcola_punti_in_obiettivo(stato: StatoPartita, colore: str) -> int:
    """
    Punti del giocatore dai suoi territori in obiettivo.
    Per ogni territorio dell'obiettivo che possiede, somma il numero
    di adiacenze di quel territorio (specifica 6.2).
    """
    giocatore = stato.giocatori[colore]
    if giocatore.obiettivo_id is None:
        return 0

    territori_target = OBIETTIVI[giocatore.obiettivo_id]["territori"]
    punti = 0
    for t in territori_target:
        if stato.mappa[t].proprietario == colore:
            punti += punti_territorio(t)
    return punti


def calcola_punti_fuori_obiettivo(stato: StatoPartita, colore: str) -> int:
    """
    Punti del giocatore dai territori che possiede ma NON sono nel suo obiettivo
    (specifica 6.3, usato per spareggio).
    """
    giocatore = stato.giocatori[colore]
    if giocatore.obiettivo_id is None:
        # Senza obiettivo, tutti i territori sono "fuori obiettivo"
        territori_target = frozenset()
    else:
        territori_target = OBIETTIVI[giocatore.obiettivo_id]["territori"]

    punti = 0
    for t, s in stato.mappa.items():
        if s.proprietario == colore and t not in territori_target:
            punti += punti_territorio(t)
    return punti


# ─────────────────────────────────────────────────────────────────────────
#  DETERMINAZIONE VINCITORE (3 criteri in cascata)
# ─────────────────────────────────────────────────────────────────────────

# Mappa colore → ordine di mano (per spareggio inverso)
ORDINE_MANO_INVERSO = {"BLU": 1, "ROSSO": 2, "VERDE": 3, "GIALLO": 4}


def determina_vincitore(stato: StatoPartita) -> str:
    """
    Determina il vincitore secondo i 3 criteri in cascata della specifica 7.3:

    Criterio 1 — Punti in obiettivo (vince chi ne ha più di tutti).
    Criterio 2 — Spareggio: punti fuori obiettivo.
    Criterio 3 — Spareggio finale: ordine di mano inverso (Giallo > Verde > Rosso > Blu).

    A ogni criterio il pool si restringe ai giocatori ancora in parità.
    Restituisce il colore del vincitore.

    Solo i giocatori VIVI partecipano al calcolo (specifica 7.4).
    """
    candidati = stato.giocatori_vivi()
    assert candidati, "Nessun giocatore vivo: situazione impossibile"

    # ── Criterio 1: punti in obiettivo ──────────────────────────────
    punti_obj = {col: calcola_punti_in_obiettivo(stato, col) for col in candidati}
    max_obj = max(punti_obj.values())
    candidati = [c for c in candidati if punti_obj[c] == max_obj]

    if len(candidati) == 1:
        return candidati[0]

    # ── Criterio 2: punti fuori obiettivo ───────────────────────────
    punti_fuori = {col: calcola_punti_fuori_obiettivo(stato, col) for col in candidati}
    max_fuori = max(punti_fuori.values())
    candidati = [c for c in candidati if punti_fuori[c] == max_fuori]

    if len(candidati) == 1:
        return candidati[0]

    # ── Criterio 3: ordine di mano inverso ──────────────────────────
    # Vince chi ha l'ordine di mano PIÙ ALTO (Giallo=4 > Verde=3 > ...)
    candidati.sort(key=lambda c: ORDINE_MANO_INVERSO[c], reverse=True)
    return candidati[0]
