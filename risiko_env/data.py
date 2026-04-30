"""
data.py — Dati statici del simulatore RisiKo.

Contiene tutte le costanti immutabili del gioco:
- 42 territori organizzati in 6 continenti
- Adiacenze (simmetriche)
- Bonus continenti
- 16 obiettivi
- Composizione del mazzo carte (44 carte: 42 territori + 2 jolly)

Specifica di riferimento: risiko_specifica_v1.2.md sezioni 2 e 6.
"""

# ─────────────────────────────────────────────────────────────────────────
#  CONTINENTI E TERRITORI
# ─────────────────────────────────────────────────────────────────────────

CONTINENTI: dict[str, list[str]] = {
    "nordamerica": [
        "alaska", "territori_del_nord_ovest", "groenlandia", "alberta",
        "ontario", "quebec", "stati_uniti_occidentali",
        "stati_uniti_orientali", "america_centrale",
    ],
    "sudamerica": [
        "venezuela", "peru", "brasile", "argentina",
    ],
    "europa": [
        "islanda", "gran_bretagna", "europa_settentrionale", "scandinavia",
        "ucraina", "europa_occidentale", "europa_meridionale",
    ],
    "africa": [
        "africa_del_nord", "egitto", "africa_orientale", "congo",
        "africa_del_sud", "madagascar",
    ],
    "asia": [
        "medio_oriente", "afghanistan", "india", "urali", "siberia",
        "jacuzia", "kamchatka", "cita", "mongolia", "cina", "siam",
        "giappone",
    ],
    "oceania": [
        "indonesia", "nuova_guinea", "australia_occidentale",
        "australia_orientale",
    ],
}

# Lista piatta di tutti i 42 territori (ordine canonico)
TUTTI_TERRITORI: list[str] = [t for terrs in CONTINENTI.values() for t in terrs]

# Mappa inversa: territorio → continente di appartenenza
CONTINENTE_DI: dict[str, str] = {
    t: cont
    for cont, terrs in CONTINENTI.items()
    for t in terrs
}

# Bonus rinforzo per continente completato
BONUS_CONTINENTE: dict[str, int] = {
    "nordamerica": 5,
    "sudamerica": 2,
    "europa": 5,
    "africa": 3,
    "asia": 7,
    "oceania": 2,
}

# ─────────────────────────────────────────────────────────────────────────
#  ADIACENZE (simmetriche)
# ─────────────────────────────────────────────────────────────────────────
# Definite "una direzione sola" qui sotto, poi il modulo le rende simmetriche
# automaticamente in fase di import (vedi fondo file).

_ADIACENZE_RAW: dict[str, list[str]] = {
    # Nord America
    "alaska": ["territori_del_nord_ovest", "alberta", "kamchatka"],
    "territori_del_nord_ovest": ["alaska", "alberta", "ontario", "groenlandia"],
    "groenlandia": ["territori_del_nord_ovest", "ontario", "quebec", "islanda"],
    "alberta": ["alaska", "territori_del_nord_ovest", "ontario", "stati_uniti_occidentali"],
    "ontario": ["territori_del_nord_ovest", "alberta", "stati_uniti_occidentali",
                "stati_uniti_orientali", "quebec", "groenlandia"],
    "quebec": ["ontario", "stati_uniti_orientali", "groenlandia"],
    "stati_uniti_occidentali": ["alberta", "ontario", "stati_uniti_orientali", "america_centrale"],
    "stati_uniti_orientali": ["stati_uniti_occidentali", "ontario", "quebec", "america_centrale"],
    "america_centrale": ["stati_uniti_occidentali", "stati_uniti_orientali", "venezuela"],

    # Sud America
    "venezuela": ["america_centrale", "peru", "brasile"],
    "peru": ["venezuela", "brasile", "argentina"],
    "brasile": ["venezuela", "peru", "argentina", "africa_del_nord"],
    "argentina": ["peru", "brasile"],

    # Europa
    "islanda": ["groenlandia", "gran_bretagna", "scandinavia"],
    "gran_bretagna": ["islanda", "europa_settentrionale", "scandinavia", "europa_occidentale"],
    "scandinavia": ["islanda", "gran_bretagna", "europa_settentrionale", "ucraina"],
    "europa_settentrionale": ["gran_bretagna", "scandinavia", "ucraina",
                              "europa_occidentale", "europa_meridionale"],
    "europa_occidentale": ["gran_bretagna", "europa_settentrionale",
                           "europa_meridionale", "africa_del_nord"],
    "europa_meridionale": ["europa_settentrionale", "europa_occidentale", "ucraina",
                           "egitto", "africa_del_nord", "medio_oriente"],
    "ucraina": ["scandinavia", "europa_settentrionale", "europa_meridionale",
                "medio_oriente", "afghanistan", "urali"],

    # Africa
    "africa_del_nord": ["europa_occidentale", "europa_meridionale", "brasile",
                        "egitto", "africa_orientale", "congo"],
    "egitto": ["europa_meridionale", "africa_del_nord", "africa_orientale", "medio_oriente"],
    "africa_orientale": ["egitto", "africa_del_nord", "congo",
                         "africa_del_sud", "madagascar"],
    "congo": ["africa_del_nord", "africa_orientale", "africa_del_sud"],
    "africa_del_sud": ["congo", "africa_orientale", "madagascar"],
    "madagascar": ["africa_orientale", "africa_del_sud"],

    # Asia
    "medio_oriente": ["europa_meridionale", "ucraina", "egitto", "afghanistan", "india"],
    "afghanistan": ["ucraina", "medio_oriente", "india", "cina", "urali"],
    "india": ["medio_oriente", "afghanistan", "cina", "siam"],
    "urali": ["ucraina", "afghanistan", "cina", "siberia"],
    "siberia": ["urali", "cina", "mongolia", "cita", "jacuzia"],
    "jacuzia": ["siberia", "cita", "kamchatka"],
    "kamchatka": ["jacuzia", "cita", "mongolia", "giappone", "alaska"],
    "cita": ["siberia", "jacuzia", "kamchatka", "mongolia"],
    "mongolia": ["cina", "siberia", "cita", "kamchatka", "giappone"],
    "cina": ["mongolia", "siberia", "urali", "afghanistan", "india", "siam"],
    "siam": ["india", "cina", "indonesia"],
    "giappone": ["kamchatka", "mongolia"],

    # Oceania
    "indonesia": ["siam", "nuova_guinea", "australia_occidentale"],
    "nuova_guinea": ["indonesia", "australia_occidentale", "australia_orientale"],
    "australia_occidentale": ["indonesia", "nuova_guinea", "australia_orientale"],
    "australia_orientale": ["nuova_guinea", "australia_occidentale"],
}


def _costruisci_adiacenze_simmetriche(raw: dict[str, list[str]]) -> dict[str, frozenset[str]]:
    """
    Rende le adiacenze simmetriche e le congela in frozenset (immutabili).
    Se A confina con B in raw, allora B confina con A nel risultato.
    """
    adj: dict[str, set[str]] = {t: set(vicini) for t, vicini in raw.items()}
    # Aggiungi simmetria
    for t, vicini in raw.items():
        for v in vicini:
            adj.setdefault(v, set()).add(t)
    # Congela in frozenset
    return {t: frozenset(vicini) for t, vicini in adj.items()}


ADIACENZE: dict[str, frozenset[str]] = _costruisci_adiacenze_simmetriche(_ADIACENZE_RAW)


def confinanti(territorio: str) -> frozenset[str]:
    """Restituisce i territori adiacenti a `territorio`."""
    return ADIACENZE[territorio]


def punti_territorio(territorio: str) -> int:
    """
    Punti del territorio = numero di territori adiacenti.
    Vedi specifica sezione 6.2.
    """
    return len(ADIACENZE[territorio])


# ─────────────────────────────────────────────────────────────────────────
#  OBIETTIVI (16 totali)
# ─────────────────────────────────────────────────────────────────────────
# Ogni obiettivo è un insieme di territori target.
# Vedi specifica sezione 6.

OBIETTIVI: dict[int, dict] = {
    1: {
        "nome": "Letto",
        "territori": frozenset([
            "alaska", "alberta", "america_centrale", "groenlandia", "ontario",
            "quebec", "stati_uniti_occidentali", "stati_uniti_orientali",
            "territori_del_nord_ovest",
            "argentina", "brasile", "peru", "venezuela",
            "australia_occidentale", "nuova_guinea", "indonesia", "siam",
            "india", "medio_oriente",
            "africa_del_nord", "congo", "egitto", "africa_orientale",
        ]),
    },
    2: {
        "nome": "Elefante",
        "territori": frozenset([
            "quebec", "groenlandia", "ontario", "islanda", "stati_uniti_orientali",
            "europa_occidentale", "europa_meridionale", "europa_settentrionale",
            "gran_bretagna", "scandinavia", "ucraina",
            "afghanistan", "urali", "medio_oriente",
            "africa_del_nord", "egitto", "congo", "africa_orientale",
            "africa_del_sud", "madagascar",
        ]),
    },
    3: {
        "nome": "Ciclista",
        "territori": frozenset([
            "europa_occidentale", "europa_meridionale", "europa_settentrionale",
            "gran_bretagna", "islanda", "scandinavia", "ucraina",
            "afghanistan", "urali", "medio_oriente", "india", "siam",
            "australia_occidentale", "australia_orientale", "nuova_guinea", "indonesia",
            "africa_del_nord", "egitto", "congo", "africa_orientale",
            "africa_del_sud", "madagascar",
        ]),
    },
    4: {
        "nome": "Giraffa",
        "territori": frozenset([
            "alaska", "alberta", "america_centrale", "groenlandia", "ontario",
            "quebec", "stati_uniti_occidentali", "stati_uniti_orientali",
            "territori_del_nord_ovest",
            "europa_meridionale", "europa_settentrionale", "gran_bretagna",
            "islanda", "scandinavia", "ucraina",
            "africa_del_nord", "egitto", "congo", "africa_orientale",
            "africa_del_sud", "madagascar",
        ]),
    },
    5: {
        "nome": "Granchio",
        "territori": frozenset([
            "alaska", "alberta", "america_centrale", "groenlandia", "ontario",
            "quebec", "stati_uniti_occidentali", "stati_uniti_orientali",
            "territori_del_nord_ovest",
            "islanda", "scandinavia", "ucraina",
            "afghanistan", "urali", "medio_oriente", "cina", "india", "siam",
            "australia_occidentale", "australia_orientale", "nuova_guinea", "indonesia",
        ]),
    },
    6: {
        "nome": "Formula1",
        "territori": frozenset([
            "europa_occidentale", "europa_meridionale", "europa_settentrionale",
            "gran_bretagna", "islanda", "scandinavia", "ucraina",
            "afghanistan", "medio_oriente", "india", "siam",
            "argentina", "brasile", "peru", "venezuela",
            "australia_occidentale", "australia_orientale", "nuova_guinea", "indonesia",
            "africa_del_nord", "egitto", "africa_orientale",
        ]),
    },
    7: {
        "nome": "Befana",
        "territori": frozenset([
            "argentina", "brasile", "peru", "venezuela",
            "africa_del_nord", "egitto", "congo", "africa_orientale",
            "africa_del_sud", "madagascar",
            "afghanistan", "urali", "medio_oriente", "india", "siam", "cina",
            "mongolia", "jacuzia", "cita", "siberia", "kamchatka", "giappone",
        ]),
    },
    8: {
        "nome": "Elvis",
        "territori": frozenset([
            "alaska", "alberta", "america_centrale", "groenlandia", "ontario",
            "quebec", "stati_uniti_occidentali", "stati_uniti_orientali",
            "territori_del_nord_ovest",
            "argentina", "brasile", "peru", "venezuela",
            "europa_occidentale", "europa_meridionale", "europa_settentrionale",
            "gran_bretagna", "islanda", "scandinavia", "ucraina",
            "kamchatka", "giappone",
        ]),
    },
    9: {
        "nome": "Dromedario con mosca",
        "territori": frozenset([
            "europa_occidentale", "europa_meridionale", "europa_settentrionale",
            "gran_bretagna", "islanda", "scandinavia", "ucraina",
            "afghanistan", "urali", "medio_oriente", "india", "siam", "cina",
            "mongolia", "jacuzia", "cita", "siberia", "kamchatka", "giappone",
            "indonesia",
        ]),
    },
    10: {
        "nome": "Piovra",
        "territori": frozenset([
            "alaska", "alberta", "america_centrale", "groenlandia", "ontario",
            "quebec", "stati_uniti_occidentali", "stati_uniti_orientali",
            "territori_del_nord_ovest",
            "europa_occidentale", "europa_meridionale", "europa_settentrionale",
            "gran_bretagna", "islanda", "scandinavia", "ucraina",
            "urali", "siberia", "kamchatka", "giappone", "jacuzia",
        ]),
    },
    11: {
        "nome": "Lupo (Siberiana)",
        "territori": frozenset([
            "europa_occidentale", "europa_meridionale", "europa_settentrionale",
            "gran_bretagna", "islanda", "scandinavia", "ucraina",
            "siberia", "urali", "afghanistan", "medio_oriente",
            "africa_del_nord", "egitto", "congo", "africa_orientale",
            "africa_del_sud", "madagascar",
            "argentina", "brasile", "peru", "venezuela",
        ]),
    },
    12: {
        "nome": "Tappeto",
        "territori": frozenset([
            "africa_del_nord", "egitto", "congo", "africa_orientale",
            "africa_del_sud", "madagascar",
            "afghanistan", "urali", "medio_oriente", "india", "siam", "cina",
            "mongolia", "jacuzia", "cita", "siberia", "kamchatka", "giappone",
            "indonesia",
            "europa_meridionale", "ucraina",
        ]),
    },
    13: {
        "nome": "Guerra fredda",
        "territori": frozenset([
            "alaska", "alberta", "america_centrale", "groenlandia", "ontario",
            "quebec", "stati_uniti_occidentali", "stati_uniti_orientali",
            "territori_del_nord_ovest",
            "afghanistan", "urali", "medio_oriente", "india", "siam", "cina",
            "mongolia", "jacuzia", "siberia", "cita", "kamchatka", "giappone",
        ]),
    },
    14: {
        "nome": "Motorino",
        "territori": frozenset([
            "argentina", "brasile", "peru", "venezuela",
            "africa_del_nord", "egitto", "congo", "africa_orientale",
            "africa_del_sud", "madagascar",
            "australia_occidentale", "australia_orientale", "nuova_guinea", "indonesia",
            "europa_occidentale", "europa_meridionale", "medio_oriente", "india",
            "siam", "cina", "mongolia", "cita", "giappone",
        ]),
    },
    15: {
        "nome": "Aragosta e pesciolino",
        "territori": frozenset([
            "alaska", "alberta",
            "egitto", "congo", "africa_orientale", "africa_del_sud", "madagascar",
            "afghanistan", "urali", "medio_oriente", "india", "siam", "cina",
            "mongolia", "jacuzia", "siberia", "kamchatka", "giappone", "cita",
            "indonesia",
            "australia_occidentale", "australia_orientale", "nuova_guinea",
        ]),
    },
    16: {
        "nome": "Locomotiva",
        "territori": frozenset([
            "alaska", "alberta", "america_centrale", "groenlandia", "ontario",
            "quebec", "stati_uniti_occidentali", "stati_uniti_orientali",
            "territori_del_nord_ovest",
            "argentina", "brasile", "peru", "venezuela",
            "africa_del_nord", "egitto", "congo", "africa_orientale",
            "africa_del_sud", "madagascar",
            "europa_occidentale", "ucraina", "europa_meridionale",
        ]),
    },
}

# ─────────────────────────────────────────────────────────────────────────
#  MAZZO CARTE
# ─────────────────────────────────────────────────────────────────────────
# 44 carte totali: 42 carte territorio + 2 jolly.
# Distribuzione simboli sulle carte territorio: 14 fanti, 14 cannoni, 14 cavalli.
# I jolly non hanno territorio né simbolo standard.
# Vedi specifica sezione 5.

# Simboli delle carte
FANTE = "fante"
CANNONE = "cannone"
CAVALLO = "cavallo"
JOLLY = "jolly"

# Distribuzione standard dei simboli sulle 42 carte territorio.
# Assegnazione fissa per riproducibilità (un territorio → sempre lo stesso simbolo).
# La sequenza ciclica fante→cannone→cavallo è una scelta arbitraria ma stabile.
def _distribuisci_simboli() -> dict[str, str]:
    """Assegna ai 42 territori un simbolo ciclando fante/cannone/cavallo."""
    simboli_ordine = [FANTE, CANNONE, CAVALLO]
    return {
        territorio: simboli_ordine[i % 3]
        for i, territorio in enumerate(TUTTI_TERRITORI)
    }


SIMBOLO_CARTA: dict[str, str] = _distribuisci_simboli()

# Numero totale di jolly nel mazzo
NUM_JOLLY = 2

# Bonus tris (vedi specifica 4.1.3)
BONUS_TRIS_3_UGUALI = 8           # 3 fanti, 3 cannoni o 3 cavalli
BONUS_TRIS_3_DIVERSI = 10         # 1 fante + 1 cannone + 1 cavallo
BONUS_TRIS_JOLLY_PIU_2 = 12       # 1 jolly + 2 carte uguali
BONUS_TERRITORIO_IN_CARTA = 2     # +2 per ogni carta del tris di territorio posseduto

# ─────────────────────────────────────────────────────────────────────────
#  PARAMETRI DI GIOCO
# ─────────────────────────────────────────────────────────────────────────

NUM_GIOCATORI = 4
COLORI_GIOCATORI = ["BLU", "ROSSO", "VERDE", "GIALLO"]  # ordine di mano

# Cap totale armate per giocatore (vedi specifica 4.1.5)
MAX_ARMATE_TOTALI = 130

# Limite carte in mano (vedi specifica 5.4)
MAX_CARTE_MANO = 7

# Soglie sdadata per round e giocatore (vedi specifica 7.2.4)
# Ritorna soglia o None se sdadata non disponibile in quel round
def soglia_sdadata(colore: str, round_n: int) -> int | None:
    """
    Restituisce la soglia di somma dadi per riuscire la sdadata.
    Vedi specifica sezione 7.2.
    """
    if colore == "GIALLO":
        if round_n < 35:
            return None
        if round_n == 35:
            return 4
        if round_n == 36:
            return 5
        if round_n == 37:
            return 6
        return 7  # round 38+
    else:
        if round_n < 36:
            return None
        if round_n == 36:
            return 4
        if round_n == 37:
            return 5
        if round_n == 38:
            return 6
        return 7  # round 39+


# Cap di sicurezza fine partita (vedi specifica 7.2.6)
ROUND_CAP_SICUREZZA = 60

# Vincolo continentale per distribuzione iniziale (vedi specifica 3.1)
def limite_continente_distribuzione(continente: str) -> int:
    """
    Massimo territori di un continente che un giocatore può ricevere
    in fase di distribuzione iniziale = floor(numero_territori / 2).
    """
    return len(CONTINENTI[continente]) // 2


# Distribuzione territori al setup (vedi specifica 3.1)
TERRITORI_PER_GIOCATORE = {
    "BLU": 10,
    "ROSSO": 10,
    "VERDE": 11,
    "GIALLO": 11,
}

# Carri da piazzare nei 7 round di piazzamento iniziale (vedi specifica 3.3)
# Indice = round (0-based, 0=primo round)
CARRI_PIAZZAMENTO_INIZIALE = {
    "BLU":    [3, 3, 3, 3, 3, 3, 2],   # totale 20
    "ROSSO":  [3, 3, 3, 3, 3, 3, 2],   # totale 20
    "VERDE":  [3, 3, 3, 3, 3, 3, 1],   # totale 19
    "GIALLO": [3, 3, 3, 3, 3, 3, 1],   # totale 19
}
