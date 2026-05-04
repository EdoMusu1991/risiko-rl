"""
opponent_tracker.py — Tracking della storia delle mosse degli avversari.

Per ogni avversario, mantiene un buffer delle ultime mosse, calcolando
statistiche aggregate (aggressività, focus su POV, expansion rate).

Usato dallo Stage A (opponent embedding) per profilare gli avversari.

Lavora con snapshot pre/post turno: confronta lo stato prima e dopo che
l'avversario gioca, per dedurre cosa ha fatto.

Estratto da env.py durante il refactoring.
"""

from .data import COLORI_GIOCATORI
from .stato import StatoPartita


# Limite dimensione storia per evitare crescita indefinita
MAX_STORIA_PER_COLORE = 50


class OpponentTracker:
    """
    Traccia le mosse degli avversari per opponent embedding.

    Uso:
        tracker = OpponentTracker(bot_color="BLU")
        # All'inizio di ogni turno avversario:
        snapshot = tracker.snapshot_pre_turno(stato, colore_avv)
        # ... avversario gioca ...
        # Dopo il turno:
        tracker.registra_mossa(stato, colore_avv, snapshot, log_callback=None)

    Dopo molti turni, `tracker.storia[colore]` contiene una lista di dict
    con statistiche di ogni mossa.
    """

    def __init__(self, bot_color: str):
        self.bot_color = bot_color
        self.storia: dict = {
            c: [] for c in COLORI_GIOCATORI if c != bot_color
        }
        # Stage A2 fix: tieni traccia dell'ultimo set di territori posseduti da
        # ciascun avversario alla FINE del suo turno precedente. Questo permette
        # di calcolare le PERDITE subite mentre era inattivo (i.e. attacchi
        # contro di lui da parte di altri).
        self._ultimo_terr_set: dict = {
            c: None for c in COLORI_GIOCATORI if c != bot_color
        }

    def reset(self) -> None:
        """Azzera tutta la storia (chiamato dal reset() dell'env)."""
        self.storia = {c: [] for c in COLORI_GIOCATORI if c != self.bot_color}
        self._ultimo_terr_set = {c: None for c in COLORI_GIOCATORI if c != self.bot_color}

    def snapshot_pre_turno(self, stato: StatoPartita, colore: str) -> dict:
        """Snapshot dello stato pre-turno avversario, da passare a registra_mossa."""
        return {
            "turno": stato.round_corrente,
            "territori_propri_pre": set(stato.territori_di(colore)),
            "territori_pov_pre": set(stato.territori_di(self.bot_color)),
            "num_attacchi_pre": (
                stato.giocatori[colore].statistiche.attacchi_totali
                if hasattr(stato.giocatori[colore], "statistiche")
                else 0
            ),
            "armate_pre": {c: stato.num_armate_di(c) for c in COLORI_GIOCATORI},
        }

    def registra_mossa(
        self,
        stato: StatoPartita,
        colore: str,
        snapshot_pre: dict,
        log_callback=None,
    ) -> None:
        """
        Confronta stato pre/post turno e aggiunge una mossa allo storico.

        Approssimazioni:
        - "attaccato" = ha conquistato territori o ne ha persi
        - "num_attacchi" = num territori cambiati di proprietà (proxy)
        - "attacchi_contro_pov" = num territori del POV ora suoi
        - "ratio_medio" = stimato come 1.5 quando attacca (proxy fisso)
        - "territori_conquistati" = aumento netto territori

        Se log_callback è dato, lo chiama per registrare l'evento.
        """
        if colore == self.bot_color or colore not in self.storia:
            return

        terr_pre = snapshot_pre["territori_propri_pre"]
        terr_post = set(stato.territori_di(colore))

        # === Conquiste durante il SUO turno ===
        # Territori che ha guadagnato attivamente
        territori_guadagnati = terr_post - terr_pre
        # Di questi guadagni, quanti erano del POV?
        territori_strappati_a_pov = (
            territori_guadagnati & snapshot_pre["territori_pov_pre"]
        )

        # === Perdite cross-turn (subite mentre era inattivo) ===
        # Confronto: territori_persi = (set fine turno scorso) - (set inizio turno corrente)
        # Questo cattura i territori che gli sono stati strappati negli intermezzi.
        ultimo = self._ultimo_terr_set.get(colore)
        if ultimo is not None:
            territori_persi_cross = ultimo - terr_pre
        else:
            territori_persi_cross = set()  # primo turno, nessun riferimento

        # Aggiorna ultimo set per il prossimo giro
        self._ultimo_terr_set[colore] = terr_post

        n_attacchi_stimati = len(territori_guadagnati)
        territori_conquistati = len(territori_guadagnati)
        attaccato = n_attacchi_stimati > 0
        attacchi_contro_pov = len(territori_strappati_a_pov)
        # Ratio medio: non calcolabile da snapshot, proxy fisso
        ratio_medio = 1.5 if attaccato else 0.0

        mossa = {
            "turno": snapshot_pre["turno"],
            "attaccato": attaccato,
            "num_attacchi": n_attacchi_stimati,
            "attacchi_contro_pov": attacchi_contro_pov,
            "ratio_medio": ratio_medio,
            "territori_conquistati": territori_conquistati,
            # Stage A2 fix: perdite calcolate fra fine turno scorso e inizio turno corrente
            # (cattura attacchi subiti negli intermezzi).
            "territori_persi": len(territori_persi_cross),
        }
        self.storia[colore].append(mossa)

        # Bound storia
        if len(self.storia[colore]) > MAX_STORIA_PER_COLORE:
            self.storia[colore] = self.storia[colore][-MAX_STORIA_PER_COLORE:]

        # Log evento se callback fornito
        if log_callback is not None:
            log_callback(
                "turno_avversario",
                colore=colore,
                attaccato=attaccato,
                territori_persi=list(territori_persi_cross),
                territori_guadagnati=list(territori_guadagnati),
                attacchi_contro_pov=attacchi_contro_pov,
            )
