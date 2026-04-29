"""
stato.py — Strutture dati per lo stato della partita.

Definisce le classi base che rappresentano lo stato del gioco:
- TerritorioStato: stato di un singolo territorio (proprietario, armate)
- Giocatore: stato di un giocatore (colore, carte, obiettivo, vivo/eliminato)
- StatoPartita: stato globale (mappa, giocatori, mazzo, round, turno corrente)

NESSUNA logica di gioco qui dentro — solo strutture dati.
La logica è nei moduli setup, motore, sdadata.

Specifica di riferimento: risiko_specifica_v1.2.md sezioni 1, 5, 8.
"""

from dataclasses import dataclass, field
from typing import Optional

from .data import (
    TUTTI_TERRITORI,
    COLORI_GIOCATORI,
    NUM_GIOCATORI,
    MAX_CARTE_MANO,
    JOLLY,
    SIMBOLO_CARTA,
)


# ─────────────────────────────────────────────────────────────────────────
#  CARTA
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Carta:
    """
    Una carta del mazzo.

    Per le carte territorio: territorio="brasile", simbolo="fante".
    Per i jolly: territorio=None, simbolo="jolly".
    """
    territorio: Optional[str]   # None per i jolly
    simbolo: str                # "fante" | "cannone" | "cavallo" | "jolly"

    @property
    def is_jolly(self) -> bool:
        return self.simbolo == JOLLY


def crea_mazzo_completo() -> list[Carta]:
    """
    Crea le 44 carte del mazzo (42 territori + 2 jolly).
    L'ordine non è mescolato — il mescolamento avviene in fase di setup.
    """
    mazzo = [
        Carta(territorio=t, simbolo=SIMBOLO_CARTA[t])
        for t in TUTTI_TERRITORI
    ]
    mazzo.append(Carta(territorio=None, simbolo=JOLLY))
    mazzo.append(Carta(territorio=None, simbolo=JOLLY))
    return mazzo


# ─────────────────────────────────────────────────────────────────────────
#  TERRITORIO
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class TerritorioStato:
    """
    Stato di un singolo territorio sulla mappa.

    Attributi:
        proprietario: colore del giocatore che lo possiede ("BLU", "ROSSO", ...)
                      None solo durante setup prima dell'assegnazione
        armate: numero di carri sul territorio (sempre >= 1 se ha proprietario)
    """
    proprietario: Optional[str] = None
    armate: int = 0


# ─────────────────────────────────────────────────────────────────────────
#  GIOCATORE
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Giocatore:
    """
    Stato di un giocatore.

    Attributi:
        colore: "BLU", "ROSSO", "VERDE", "GIALLO"
        ordine_mano: 1 (Blu), 2 (Rosso), 3 (Verde), 4 (Giallo)
        obiettivo_id: ID dell'obiettivo segreto (1-16). None solo durante setup.
        carte: lista di carte in mano (max MAX_CARTE_MANO=7)
        vivo: True se è ancora in partita, False se eliminato
    """
    colore: str
    ordine_mano: int
    obiettivo_id: Optional[int] = None
    carte: list[Carta] = field(default_factory=list)
    vivo: bool = True

    def num_carte(self) -> int:
        return len(self.carte)

    def puo_pescare(self) -> bool:
        """Vedi specifica 4.4: non si può pescare se a 7 carte."""
        return self.num_carte() < MAX_CARTE_MANO


# ─────────────────────────────────────────────────────────────────────────
#  STATO GLOBALE PARTITA
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class StatoPartita:
    """
    Stato globale della partita.

    Contiene:
    - mappa: dict territorio → TerritorioStato
    - giocatori: dict colore → Giocatore
    - mazzo_attivo: carte da pescare
    - pila_scarti: carte usate (tris giocati, eccedenze post-eliminazione)
    - round_corrente: numero del round (1+)
    - giocatore_corrente: colore di chi sta giocando ora (None se partita non iniziata)
    - conquiste_turno_corrente: dict colore → quanti territori conquistati nel turno
                                (resettato a inizio di ogni turno)
    - vincitore: colore del vincitore, oppure None se partita in corso
    - terminata: True se la partita è finita
    - motivo_fine: stringa descrittiva ("obiettivo_completato", "sdadata",
                                       "cap_sicurezza", None)
    """
    mappa: dict[str, TerritorioStato] = field(default_factory=dict)
    giocatori: dict[str, Giocatore] = field(default_factory=dict)
    mazzo_attivo: list[Carta] = field(default_factory=list)
    pila_scarti: list[Carta] = field(default_factory=list)

    round_corrente: int = 0
    giocatore_corrente: Optional[str] = None
    conquiste_turno_corrente: dict[str, int] = field(default_factory=dict)

    vincitore: Optional[str] = None
    terminata: bool = False
    motivo_fine: Optional[str] = None

    def __post_init__(self):
        # Inizializza la mappa con tutti i 42 territori vuoti
        if not self.mappa:
            self.mappa = {t: TerritorioStato() for t in TUTTI_TERRITORI}

        # Inizializza i 4 giocatori con i colori canonici
        if not self.giocatori:
            for ordine, colore in enumerate(COLORI_GIOCATORI, start=1):
                self.giocatori[colore] = Giocatore(
                    colore=colore,
                    ordine_mano=ordine,
                )

    # ─────────────────────────────────────────────────────────────
    #  Helper di lettura (no logica, solo accessi)
    # ─────────────────────────────────────────────────────────────

    def territori_di(self, colore: str) -> list[str]:
        """Lista dei territori posseduti dal giocatore."""
        return [t for t, s in self.mappa.items() if s.proprietario == colore]

    def num_territori_di(self, colore: str) -> int:
        """Numero di territori posseduti dal giocatore."""
        return sum(1 for s in self.mappa.values() if s.proprietario == colore)

    def num_armate_di(self, colore: str) -> int:
        """Totale armate del giocatore sulla mappa."""
        return sum(s.armate for s in self.mappa.values() if s.proprietario == colore)

    def giocatori_vivi(self) -> list[str]:
        """Lista colori dei giocatori ancora in partita, in ordine di mano."""
        return [
            g.colore for g in sorted(self.giocatori.values(),
                                     key=lambda g: g.ordine_mano)
            if g.vivo
        ]

    def giocatore(self, colore: str) -> Giocatore:
        """Shortcut per accedere a un giocatore."""
        return self.giocatori[colore]


# ─────────────────────────────────────────────────────────────────────────
#  ENUM FASI
# ─────────────────────────────────────────────────────────────────────────
# Le 4 fasi del turno (vedi specifica sezione 4)

class Fase:
    """Costanti per le fasi del turno (alternativa più leggera a Enum)."""
    TRIS_E_RINFORZI = "TRIS_E_RINFORZI"
    ATTACCHI = "ATTACCHI"
    SPOSTAMENTO = "SPOSTAMENTO"
    PESCA_CARTA = "PESCA_CARTA"

    ORDINE = [TRIS_E_RINFORZI, ATTACCHI, SPOSTAMENTO, PESCA_CARTA]
