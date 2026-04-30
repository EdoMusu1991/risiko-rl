"""
event_log.py — Sistema di logging degli eventi della partita.

Buffer leggero per registrare eventi (attacchi, rinforzi, conquiste, ecc.)
durante una partita. Usato dal visualizzatore.

Quando log_eventi=False (default in training), questo logger è no-op:
zero overhead.

Estratto da env.py durante il refactoring.
"""


class EventLogger:
    """
    Logger di eventi della partita.

    Uso:
        logger = EventLogger(attivo=True)
        logger.log("attacco_bot", da="alaska", verso="alberta", conquistato=True)
        # ...
        for e in logger.eventi:
            print(e)
    """

    def __init__(self, attivo: bool = False):
        self.attivo = attivo
        self.eventi: list = []

    def reset(self) -> None:
        """Azzera la lista eventi."""
        self.eventi = []

    def log(self, tipo: str, round: int = 0, turno_di: str = None, **dati) -> None:
        """Registra un evento (no-op se logger non attivo)."""
        if not self.attivo:
            return
        evento = {
            "tipo": tipo,
            "round": round,
            "turno_di": turno_di,
            **dati,
        }
        self.eventi.append(evento)
