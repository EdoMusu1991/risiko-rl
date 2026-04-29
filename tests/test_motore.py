"""
test_motore.py — Test del Modulo 3 (motore di gioco).

Verifica:
- Combattimento: numero dadi, lancio, risoluzione coppie, parità al difensore
- Calcolo rinforzi: formula base, bonus continenti
- Tris: rilevamento combinazioni, bonus territori
- Attacchi: legalità, conquista, catena, eliminazione
- Spostamento finale: regole nicchia/non-nicchia
- Pesca carta: conquiste, limite 7, reshuffling
- Vittoria immediata per obiettivo completato
- Avanzamento turno e round

Esegui: python tests\test_motore.py
"""

import sys
import os
import random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env.data import (
    OBIETTIVI,
    COLORI_GIOCATORI,
    FANTE,
    CANNONE,
    CAVALLO,
    JOLLY,
)
from risiko_env.stato import StatoPartita, Carta, TerritorioStato
from risiko_env.setup import crea_partita_iniziale
from risiko_env.combattimento import (
    num_dadi_attaccante,
    num_dadi_difensore,
    lancia_dadi,
    risolvi_lancio,
)
from risiko_env.motore import (
    calcola_rinforzi_base,
    calcola_bonus_continenti,
    trova_tutti_i_tris,
    seleziona_due_tris_disgiunti,
    calcola_bonus_tris,
    gioca_tris,
    piazza_rinforzi,
    attacco_legale,
    territori_attaccabili_da,
    esegui_attacco,
    applica_conquista,
    spostamento_legale,
    esegui_spostamento,
    pesca_carta,
    prossimo_giocatore,
    avanza_turno,
)
from risiko_env.obiettivi import (
    ha_completato_obiettivo,
    calcola_punti_in_obiettivo,
    calcola_punti_fuori_obiettivo,
    determina_vincitore,
)


# ═════════════════════════════════════════════════════════════════════════
#  COMBATTIMENTO
# ═════════════════════════════════════════════════════════════════════════

def test_num_dadi_attaccante():
    assert num_dadi_attaccante(1) == 0  # Non può attaccare
    assert num_dadi_attaccante(2) == 1
    assert num_dadi_attaccante(3) == 2
    assert num_dadi_attaccante(4) == 3
    assert num_dadi_attaccante(10) == 3  # Massimo 3
    print("✓ num_dadi_attaccante: 1→0, 2→1, 3→2, 4+→3")


def test_num_dadi_difensore():
    assert num_dadi_difensore(0) == 0
    assert num_dadi_difensore(1) == 1
    assert num_dadi_difensore(2) == 2
    assert num_dadi_difensore(3) == 3
    assert num_dadi_difensore(10) == 3  # Massimo 3
    print("✓ num_dadi_difensore: 0→0, 1→1, 2→2, 3+→3")


def test_lancia_dadi_decrescenti():
    """Verifica che lancia_dadi restituisca sempre dadi ordinati decrescenti."""
    rng = random.Random(42)
    for _ in range(100):
        for n in [1, 2, 3]:
            dadi = lancia_dadi(n, rng)
            assert len(dadi) == n
            assert all(1 <= d <= 6 for d in dadi)
            assert dadi == sorted(dadi, reverse=True)
    print("✓ lancia_dadi sempre ordinati decrescenti, valori 1-6")


def test_risolvi_lancio_parita_a_difensore():
    """Test critico: in caso di parità, vince il difensore."""
    # Forziamo parità tirando dadi controllati con un mock RNG
    class FakeRng:
        def __init__(self, vals):
            self.vals = list(vals)
            self.idx = 0
        def randint(self, a, b):
            v = self.vals[self.idx]
            self.idx += 1
            return v

    # Attacco 3v3, dadi: att=[5,4,3], dif=[5,4,3] → 3 parità → att perde 3
    rng = FakeRng([5, 4, 3, 5, 4, 3])
    perdite_att, perdite_dif, _, _ = risolvi_lancio(4, 3, rng)
    assert perdite_att == 3, f"Attaccante doveva perdere 3, perse {perdite_att}"
    assert perdite_dif == 0, f"Difensore non doveva perdere, perse {perdite_dif}"
    print("✓ Parità → difensore vince (test su 3 parità consecutive)")


def test_risolvi_lancio_attaccante_vince():
    """Attaccante con dadi superiori vince."""
    class FakeRng:
        def __init__(self, vals):
            self.vals = list(vals)
            self.idx = 0
        def randint(self, a, b):
            v = self.vals[self.idx]
            self.idx += 1
            return v

    # Attacco 4v3 (att 3 dadi, dif 3 dadi), att=[6,6,6], dif=[1,1,1] → att vince 3
    rng = FakeRng([6, 6, 6, 1, 1, 1])
    perdite_att, perdite_dif, _, _ = risolvi_lancio(4, 3, rng)
    assert perdite_att == 0
    assert perdite_dif == 3
    print("✓ Attaccante con dadi superiori vince tutto")


def test_risolvi_lancio_min_coppie():
    """Si confrontano min(num_att, num_dif) coppie."""
    class FakeRng:
        def __init__(self, vals):
            self.vals = list(vals)
            self.idx = 0
        def randint(self, a, b):
            v = self.vals[self.idx]
            self.idx += 1
            return v

    # 3v1: att tira 3 dadi, dif tira 1 dado, si confronta solo 1 coppia
    # att=[5,4,3], dif=[6] → 6>5 → att perde 1, dif perde 0
    rng = FakeRng([3, 4, 5, 6])  # randint chiamata 4 volte (3 att + 1 dif)
    perdite_att, perdite_dif, _, _ = risolvi_lancio(4, 1, rng)
    assert perdite_att == 1
    assert perdite_dif == 0
    print("✓ Solo min(att, dif) coppie confrontate (3v1 = 1 confronto)")


# ═════════════════════════════════════════════════════════════════════════
#  RINFORZI
# ═════════════════════════════════════════════════════════════════════════

def test_rinforzi_base():
    """Rinforzi = floor(territori / 3), nessun minimo."""
    stato = StatoPartita()
    # 0 territori → 0 rinforzi
    assert calcola_rinforzi_base(stato, "BLU") == 0
    # Assegna 5 territori
    territori_test = ["alaska", "alberta", "ontario", "quebec", "groenlandia"]
    for t in territori_test:
        stato.mappa[t].proprietario = "BLU"
        stato.mappa[t].armate = 1
    assert calcola_rinforzi_base(stato, "BLU") == 1  # 5//3 = 1

    # 12 territori → 4 rinforzi
    altri = ["islanda", "scandinavia", "ucraina", "europa_occidentale",
             "africa_del_nord", "egitto", "venezuela"]
    for t in altri:
        stato.mappa[t].proprietario = "BLU"
        stato.mappa[t].armate = 1
    assert calcola_rinforzi_base(stato, "BLU") == 4  # 12//3 = 4

    print("✓ Rinforzi base: 5→1, 12→4 (floor(t/3) senza minimo)")


def test_bonus_continente():
    """Possedere un intero continente dà il bonus."""
    stato = StatoPartita()
    # Assegna tutta l'Oceania (4 territori) a Blu
    oceania = ["indonesia", "nuova_guinea", "australia_occidentale", "australia_orientale"]
    for t in oceania:
        stato.mappa[t].proprietario = "BLU"
        stato.mappa[t].armate = 1
    assert calcola_bonus_continenti(stato, "BLU") == 2  # Oceania = +2

    # Aggiungi Sud America (+2)
    sudamerica = ["venezuela", "peru", "brasile", "argentina"]
    for t in sudamerica:
        stato.mappa[t].proprietario = "BLU"
        stato.mappa[t].armate = 1
    assert calcola_bonus_continenti(stato, "BLU") == 4  # OC + SA = 2 + 2

    print("✓ Bonus continenti: Oceania=+2, +SA=+4")


# ═════════════════════════════════════════════════════════════════════════
#  TRIS
# ═════════════════════════════════════════════════════════════════════════

def test_tris_3_uguali():
    """3 fanti = tris da 8 armate."""
    carte = [
        Carta("alaska", FANTE),
        Carta("brasile", FANTE),
        Carta("cina", FANTE),
    ]
    tris = trova_tutti_i_tris(carte)
    assert len(tris) == 1
    assert tris[0][1] == 8  # bonus 3 uguali
    print("✓ Tris 3 uguali = +8")


def test_tris_3_diversi():
    """1 fante + 1 cannone + 1 cavallo = tris da 10."""
    carte = [
        Carta("alaska", FANTE),
        Carta("brasile", CANNONE),
        Carta("cina", CAVALLO),
    ]
    tris = trova_tutti_i_tris(carte)
    assert len(tris) == 1
    assert tris[0][1] == 10
    print("✓ Tris 3 diversi (F+C+Cav) = +10")


def test_tris_jolly_piu_2():
    """1 jolly + 2 uguali = tris da 12."""
    carte = [
        Carta(None, JOLLY),
        Carta("alaska", FANTE),
        Carta("brasile", FANTE),
    ]
    tris = trova_tutti_i_tris(carte)
    assert len(tris) == 1
    assert tris[0][1] == 12
    print("✓ Tris jolly+2uguali = +12")


def test_seleziona_2_tris_disgiunti():
    """Con 6 carte ci possono essere 2 tris disgiunti."""
    carte = [
        Carta("alaska", FANTE),
        Carta("brasile", FANTE),
        Carta("cina", FANTE),  # 3 fanti = tris 1
        Carta("egitto", CANNONE),
        Carta("india", CAVALLO),
        Carta("ucraina", FANTE),  # 3 diversi: F + C + Cav?
    ]
    # In realtà qui abbiamo 4 fanti + 1 cannone + 1 cavallo
    # Il primo tris (3 uguali) prende 3 fanti → restano 1 fante + 1 cannone + 1 cavallo
    # Secondo tris: 3 diversi
    selezionati = seleziona_due_tris_disgiunti(carte)
    assert len(selezionati) == 2, f"Trovati {len(selezionati)} tris invece di 2"
    print(f"✓ Selezione 2 tris disgiunti: {selezionati[0][1]} + {selezionati[1][1]} armate")


def test_bonus_territorio_in_carta():
    """+2 armate per ogni carta del tris di territorio posseduto."""
    stato = StatoPartita()
    stato.mappa["alaska"].proprietario = "BLU"
    stato.mappa["alaska"].armate = 1
    stato.mappa["brasile"].proprietario = "BLU"
    stato.mappa["brasile"].armate = 1
    # NB: cina NON è di Blu

    carte = [
        Carta("alaska", FANTE),
        Carta("brasile", FANTE),
        Carta("cina", FANTE),
    ]
    # Tris 3 uguali = +8, di cui alaska e brasile sono di Blu → +2+2 = +4
    # Totale atteso: 8 + 4 = 12
    tris_giocati = [(carte, 8)]
    bonus = calcola_bonus_tris(stato, "BLU", tris_giocati)
    assert bonus == 12, f"Atteso 12, ottenuto {bonus}"
    print("✓ Bonus territori (alaska+brasile posseduti) = 8+4=12")


# ═════════════════════════════════════════════════════════════════════════
#  PIAZZAMENTO RINFORZI
# ═════════════════════════════════════════════════════════════════════════

def test_piazza_rinforzi_normale():
    stato = crea_partita_iniziale(seed=42)
    # Prendi un territorio di Blu
    territori_blu = stato.territori_di("BLU")
    t = territori_blu[0]
    armate_prima = stato.mappa[t].armate
    piazzate = piazza_rinforzi(stato, "BLU", {t: 5})
    assert piazzate == 5
    assert stato.mappa[t].armate == armate_prima + 5
    print("✓ piazza_rinforzi normale (5 armate aggiunte)")


def test_piazza_rinforzi_cap_130():
    """Il cap 130 limita le armate aggiunte."""
    stato = StatoPartita()
    # Setup minimale: 1 territorio a Blu con 128 armate
    stato.mappa["alaska"].proprietario = "BLU"
    stato.mappa["alaska"].armate = 128
    # Provo a piazzare 10 armate → cap 130, ne entrano solo 2
    piazzate = piazza_rinforzi(stato, "BLU", {"alaska": 10})
    assert piazzate == 2
    assert stato.mappa["alaska"].armate == 130
    print("✓ Cap 130: 10 richieste, 2 piazzate (132→130)")


# ═════════════════════════════════════════════════════════════════════════
#  ATTACCO E CONQUISTA
# ═════════════════════════════════════════════════════════════════════════

def test_attacco_legale():
    stato = StatoPartita()
    stato.mappa["brasile"].proprietario = "BLU"
    stato.mappa["brasile"].armate = 5
    stato.mappa["argentina"].proprietario = "ROSSO"
    stato.mappa["argentina"].armate = 1

    assert attacco_legale(stato, "BLU", "brasile", "argentina")
    # Auto-attacco illegale
    stato.mappa["peru"].proprietario = "BLU"
    stato.mappa["peru"].armate = 3
    assert not attacco_legale(stato, "BLU", "brasile", "peru")
    # Non adiacente
    assert not attacco_legale(stato, "BLU", "brasile", "alaska")
    # Senza armate
    stato.mappa["brasile"].armate = 1
    assert not attacco_legale(stato, "BLU", "brasile", "argentina")
    print("✓ attacco_legale: corretto, auto-attacco no, non adiacente no, 1 armata no")


def test_attacco_completo_con_conquista():
    """Attacco con vantaggio enorme deve quasi sempre conquistare."""
    stato = StatoPartita()
    stato.mappa["brasile"].proprietario = "BLU"
    stato.mappa["brasile"].armate = 20
    stato.mappa["argentina"].proprietario = "ROSSO"
    stato.mappa["argentina"].armate = 1
    # Inizializza giocatori vivi e altri territori per non eliminare ROSSO
    stato.mappa["peru"].proprietario = "ROSSO"
    stato.mappa["peru"].armate = 1

    # Verifica preliminare
    rng = random.Random(42)
    esito = esegui_attacco(stato, "BLU", "brasile", "argentina", rng)
    assert esito.conquistato
    assert stato.mappa["argentina"].armate == 0
    # Sposta 3 armate (numero dadi ultimo lancio = 3)
    minimo = esito.num_dadi_ultimo_lancio
    massimo = stato.mappa["brasile"].armate - 1
    fine = applica_conquista(stato, "BLU", "brasile", "argentina",
                             minimo, esito, rng)
    assert not fine  # non è vittoria immediata
    assert stato.mappa["argentina"].proprietario == "BLU"
    assert stato.mappa["argentina"].armate == minimo
    assert stato.conquiste_turno_corrente.get("BLU", 0) == 1
    print("✓ Attacco con conquista: territorio cambia proprietario, armate spostate, contatore +1")


def test_eliminazione_giocatore_ruba_carte():
    """Eliminare un giocatore ruba le sue carte."""
    stato = StatoPartita()
    stato.mappa["alaska"].proprietario = "BLU"
    stato.mappa["alaska"].armate = 10
    stato.mappa["territori_del_nord_ovest"].proprietario = "ROSSO"
    stato.mappa["territori_del_nord_ovest"].armate = 1
    # ROSSO ha solo questo territorio → sarà eliminato
    # Aggiungi 3 carte alla mano di Rosso
    stato.giocatori["ROSSO"].carte = [
        Carta("brasile", FANTE),
        Carta("egitto", CANNONE),
        Carta("cina", CAVALLO),
    ]
    # Assegna obiettivo a entrambi (richiesto dal motore)
    stato.giocatori["BLU"].obiettivo_id = 1
    stato.giocatori["ROSSO"].obiettivo_id = 2

    rng = random.Random(42)
    esito = esegui_attacco(stato, "BLU", "alaska",
                          "territori_del_nord_ovest", rng)
    assert esito.conquistato
    applica_conquista(stato, "BLU", "alaska", "territori_del_nord_ovest",
                     esito.num_dadi_ultimo_lancio, esito, rng)

    # Rosso eliminato
    assert not stato.giocatori["ROSSO"].vivo
    assert len(stato.giocatori["ROSSO"].carte) == 0
    # Blu ha ricevuto le 3 carte
    assert len(stato.giocatori["BLU"].carte) == 3
    print("✓ Eliminazione: Rosso vivo=False, 3 carte trasferite a Blu")


def test_vittoria_immediata_per_obiettivo():
    """Conquistando l'ultimo territorio dell'obiettivo → vittoria immediata."""
    stato = StatoPartita()
    # Obiettivo "Letto" (id=1) ha 23 territori
    obj_id = 1
    territori_obj = OBIETTIVI[obj_id]["territori"]

    # Assegna tutti i territori dell'obiettivo a Blu, tranne 1 a Rosso
    territori_lista = list(territori_obj)
    for t in territori_lista[:-1]:
        stato.mappa[t].proprietario = "BLU"
        stato.mappa[t].armate = 5
    # L'ultimo territorio è di Rosso
    ultimo_territorio = territori_lista[-1]
    stato.mappa[ultimo_territorio].proprietario = "ROSSO"
    stato.mappa[ultimo_territorio].armate = 1

    # Setup attaccante: serve un territorio di Blu adiacente all'ultimo_territorio
    # Trova un Blu adiacente
    from risiko_env.data import ADIACENZE
    adiacenti_obj = ADIACENZE[ultimo_territorio]
    adiacente_di_blu = None
    for adj in adiacenti_obj:
        if stato.mappa[adj].proprietario == "BLU":
            adiacente_di_blu = adj
            stato.mappa[adj].armate = 20  # rinforza per garantire vittoria
            break
    if adiacente_di_blu is None:
        # Caso raro: forza un territorio adiacente non in obiettivo
        adj = next(iter(adiacenti_obj))
        stato.mappa[adj].proprietario = "BLU"
        stato.mappa[adj].armate = 20
        adiacente_di_blu = adj

    # Aggiungi un altro territorio a Rosso per non eliminarlo (così la vittoria
    # è veramente per obiettivo, non per eliminazione)
    rosso_extra = None
    for t, s in stato.mappa.items():
        if s.proprietario is None:
            stato.mappa[t].proprietario = "ROSSO"
            stato.mappa[t].armate = 1
            rosso_extra = t
            break

    stato.giocatori["BLU"].obiettivo_id = obj_id
    stato.giocatori["ROSSO"].obiettivo_id = 2

    rng = random.Random(42)
    esito = esegui_attacco(stato, "BLU", adiacente_di_blu, ultimo_territorio, rng)
    assert esito.conquistato
    fine = applica_conquista(stato, "BLU", adiacente_di_blu, ultimo_territorio,
                            esito.num_dadi_ultimo_lancio, esito, rng)
    assert fine, "Doveva tornare True (vittoria immediata)"
    assert stato.terminata
    assert stato.vincitore == "BLU"
    assert stato.motivo_fine == "obiettivo_completato"
    assert ha_completato_obiettivo(stato, "BLU")
    print(f"✓ Vittoria immediata per obiettivo completato (obj={obj_id} 'Letto')")


# ═════════════════════════════════════════════════════════════════════════
#  SPOSTAMENTO
# ═════════════════════════════════════════════════════════════════════════

def test_spostamento_legale_nicchia():
    """In nicchia (tutti adiacenti propri): minimo da lasciare = 1."""
    stato = StatoPartita()
    # Tutto Nord America di Blu
    from risiko_env.data import CONTINENTI
    for t in CONTINENTI["nordamerica"]:
        stato.mappa[t].proprietario = "BLU"
        stato.mappa[t].armate = 5

    # alaska in nicchia? Vediamo i suoi adiacenti
    # Adiacenti di alaska: territori_del_nord_ovest (Blu), alberta (Blu), kamchatka
    # kamchatka non è in NA, quindi NON è Blu → alaska NON è in nicchia
    # Per testare nicchia uso un territorio con TUTTI adiacenti in NA
    # ontario ha adiacenti: territori_nord_ovest, alberta, stati_occ, stati_or, quebec, groenlandia
    # Tutti in NA → ontario è in nicchia se NA è tutto Blu
    assert spostamento_legale(stato, "BLU", "ontario", "alberta", 4)  # ok lascia 1
    assert not spostamento_legale(stato, "BLU", "ontario", "alberta", 5)  # lascerebbe 0
    print("✓ Spostamento da nicchia: ok lasciare 1, no lasciare 0")


def test_spostamento_non_nicchia():
    """Se ha confinanti nemici: minimo da lasciare = 2."""
    stato = StatoPartita()
    stato.mappa["brasile"].proprietario = "BLU"
    stato.mappa["brasile"].armate = 5
    stato.mappa["argentina"].proprietario = "BLU"
    stato.mappa["argentina"].armate = 1
    # peru è di Rosso → brasile non è in nicchia
    stato.mappa["peru"].proprietario = "ROSSO"
    stato.mappa["peru"].armate = 1
    # Sposto da brasile a argentina, max = 5 - 2 = 3
    assert spostamento_legale(stato, "BLU", "brasile", "argentina", 3)  # ok lascia 2
    assert not spostamento_legale(stato, "BLU", "brasile", "argentina", 4)  # lascerebbe 1
    print("✓ Spostamento non in nicchia: ok lasciare 2, no lasciare 1")


def test_esegui_spostamento():
    stato = StatoPartita()
    stato.mappa["brasile"].proprietario = "BLU"
    stato.mappa["brasile"].armate = 5
    stato.mappa["argentina"].proprietario = "BLU"
    stato.mappa["argentina"].armate = 1
    stato.mappa["peru"].proprietario = "ROSSO"
    stato.mappa["peru"].armate = 1

    esegui_spostamento(stato, "BLU", "brasile", "argentina", 3)
    assert stato.mappa["brasile"].armate == 2
    assert stato.mappa["argentina"].armate == 4
    print("✓ esegui_spostamento aggiorna correttamente armate")


# ═════════════════════════════════════════════════════════════════════════
#  PESCA CARTA
# ═════════════════════════════════════════════════════════════════════════

def test_pesca_carta_se_ha_conquistato():
    stato = crea_partita_iniziale(seed=42)
    rng = random.Random(42)
    # Marca conquista per Blu
    stato.conquiste_turno_corrente["BLU"] = 1
    carte_prima = len(stato.giocatori["BLU"].carte)
    pescato = pesca_carta(stato, "BLU", rng)
    assert pescato
    assert len(stato.giocatori["BLU"].carte) == carte_prima + 1
    print("✓ Pesca carta dopo conquista")


def test_no_pesca_se_non_conquistato():
    stato = crea_partita_iniziale(seed=42)
    rng = random.Random(42)
    stato.conquiste_turno_corrente["BLU"] = 0  # niente conquiste
    pescato = pesca_carta(stato, "BLU", rng)
    assert not pescato
    print("✓ No pesca senza conquista")


def test_no_pesca_se_7_carte():
    stato = crea_partita_iniziale(seed=42)
    # Riempi a 7 carte
    stato.giocatori["BLU"].carte = [Carta("alaska", FANTE) for _ in range(7)]
    stato.conquiste_turno_corrente["BLU"] = 1
    rng = random.Random(42)
    pescato = pesca_carta(stato, "BLU", rng)
    assert not pescato
    assert len(stato.giocatori["BLU"].carte) == 7
    print("✓ No pesca se già a 7 carte (limite inviolabile)")


# ═════════════════════════════════════════════════════════════════════════
#  OBIETTIVI E PUNTEGGI
# ═════════════════════════════════════════════════════════════════════════

def test_calcola_punti_in_obiettivo():
    stato = StatoPartita()
    obj_id = 16  # Locomotiva
    stato.giocatori["BLU"].obiettivo_id = obj_id
    territori_obj = list(OBIETTIVI[obj_id]["territori"])

    # Assegna 3 territori dell'obiettivo a Blu
    for t in territori_obj[:3]:
        stato.mappa[t].proprietario = "BLU"
        stato.mappa[t].armate = 1

    from risiko_env.data import punti_territorio
    atteso = sum(punti_territorio(t) for t in territori_obj[:3])
    actual = calcola_punti_in_obiettivo(stato, "BLU")
    assert actual == atteso, f"Atteso {atteso}, ottenuto {actual}"
    print(f"✓ Punti in obiettivo: {actual} (3 territori posseduti)")


def test_determina_vincitore_caso_specifica():
    """
    Esempio dalla specifica 7.3:
    Rosso 43 / Blu 43 / Verde 40 / Giallo 20 (tutti con 0 fuori obiettivo).
    Vincitore: Rosso (criterio 3, ordine inverso, Rosso > Blu).
    """
    stato = StatoPartita()
    # Setup obiettivi e territori per simulare i punteggi
    # Useremo manipolazione diretta per controllare

    # Mock: assegniamo gli obiettivi e i territori in modo da ottenere quei punti
    # Per semplicità, sovrascrivo le funzioni di punteggio in un test di alto livello
    # Qui simulo con un setup artificiale

    # Soluzione: dò a Rosso un obiettivo grosso e tanti territori dell'obiettivo
    # idem Blu, con stesso punteggio. Verde meno. Giallo poco.
    # Visto che è complesso, faccio un test che verifica solo la cascata di criteri:

    # Useremo monkey-patching delle funzioni di punteggio per il test
    import risiko_env.obiettivi as obj_module
    # Salva originali
    orig_in = obj_module.calcola_punti_in_obiettivo
    orig_fuori = obj_module.calcola_punti_fuori_obiettivo

    # Mock: punti fissi
    punti_in = {"BLU": 43, "ROSSO": 43, "VERDE": 40, "GIALLO": 20}
    punti_fuori = {"BLU": 0, "ROSSO": 0, "VERDE": 0, "GIALLO": 0}
    obj_module.calcola_punti_in_obiettivo = lambda s, c: punti_in[c]
    obj_module.calcola_punti_fuori_obiettivo = lambda s, c: punti_fuori[c]

    try:
        vincitore = determina_vincitore(stato)
        assert vincitore == "ROSSO", f"Atteso ROSSO, ottenuto {vincitore}"
    finally:
        obj_module.calcola_punti_in_obiettivo = orig_in
        obj_module.calcola_punti_fuori_obiettivo = orig_fuori

    print("✓ Cascata criteri: Rosso 43 / Blu 43 / V 40 / G 20 → vince Rosso (ord. inverso vs Blu)")


def test_determina_vincitore_dominante():
    """Se uno è dominante al criterio 1, vince subito."""
    stato = StatoPartita()
    import risiko_env.obiettivi as obj_module
    orig_in = obj_module.calcola_punti_in_obiettivo
    obj_module.calcola_punti_in_obiettivo = lambda s, c: {"BLU": 50, "ROSSO": 30,
                                                         "VERDE": 20, "GIALLO": 10}[c]
    try:
        vincitore = determina_vincitore(stato)
        assert vincitore == "BLU"
    finally:
        obj_module.calcola_punti_in_obiettivo = orig_in
    print("✓ Cascata criteri: dominante al criterio 1 vince subito")


# ═════════════════════════════════════════════════════════════════════════
#  AVANZAMENTO TURNO
# ═════════════════════════════════════════════════════════════════════════

def test_avanza_turno_normale():
    """BLU → ROSSO → VERDE → GIALLO → BLU (round +1)."""
    stato = crea_partita_iniziale(seed=42)
    assert stato.giocatore_corrente == "BLU"
    assert stato.round_corrente == 1

    avanza_turno(stato)
    assert stato.giocatore_corrente == "ROSSO"
    assert stato.round_corrente == 1

    avanza_turno(stato)
    assert stato.giocatore_corrente == "VERDE"
    avanza_turno(stato)
    assert stato.giocatore_corrente == "GIALLO"

    avanza_turno(stato)
    assert stato.giocatore_corrente == "BLU"
    assert stato.round_corrente == 2

    print("✓ Avanzamento normale: BLU→R→V→G→BLU(round+1)")


def test_avanza_turno_con_eliminato():
    """Se un giocatore è eliminato, viene saltato."""
    stato = crea_partita_iniziale(seed=42)
    # Elimina Rosso
    stato.giocatori["ROSSO"].vivo = False

    avanza_turno(stato)
    # Da BLU dovrebbe saltare a VERDE (Rosso eliminato)
    assert stato.giocatore_corrente == "VERDE", f"Atteso VERDE, ottenuto {stato.giocatore_corrente}"
    print("✓ Avanzamento salta giocatori eliminati")


# ═════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═════════════════════════════════════════════════════════════════════════

def main():
    tests = [
        # Combattimento
        test_num_dadi_attaccante,
        test_num_dadi_difensore,
        test_lancia_dadi_decrescenti,
        test_risolvi_lancio_parita_a_difensore,
        test_risolvi_lancio_attaccante_vince,
        test_risolvi_lancio_min_coppie,
        # Rinforzi
        test_rinforzi_base,
        test_bonus_continente,
        # Tris
        test_tris_3_uguali,
        test_tris_3_diversi,
        test_tris_jolly_piu_2,
        test_seleziona_2_tris_disgiunti,
        test_bonus_territorio_in_carta,
        # Piazzamento
        test_piazza_rinforzi_normale,
        test_piazza_rinforzi_cap_130,
        # Attacchi
        test_attacco_legale,
        test_attacco_completo_con_conquista,
        test_eliminazione_giocatore_ruba_carte,
        test_vittoria_immediata_per_obiettivo,
        # Spostamento
        test_spostamento_legale_nicchia,
        test_spostamento_non_nicchia,
        test_esegui_spostamento,
        # Pesca carta
        test_pesca_carta_se_ha_conquistato,
        test_no_pesca_se_non_conquistato,
        test_no_pesca_se_7_carte,
        # Obiettivi e vincitore
        test_calcola_punti_in_obiettivo,
        test_determina_vincitore_caso_specifica,
        test_determina_vincitore_dominante,
        # Avanzamento turno
        test_avanza_turno_normale,
        test_avanza_turno_con_eliminato,
    ]

    print("\n" + "=" * 60)
    print("Test Modulo 3: Motore di gioco")
    print("=" * 60 + "\n")

    falliti = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} FALLITO: {e}")
            falliti.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} ERRORE: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            falliti.append(test.__name__)

    print("\n" + "=" * 60)
    if falliti:
        print(f"FALLITI: {len(falliti)}/{len(tests)}")
        for nome in falliti:
            print(f"  - {nome}")
    else:
        print(f"TUTTI I {len(tests)} TEST PASSATI ✓")
    print("=" * 60 + "\n")

    return len(falliti) == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
