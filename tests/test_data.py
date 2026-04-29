"""
test_data.py — Test per il Modulo 1 (dati statici e strutture base).

Verifica:
- 42 territori esatti, distribuiti correttamente nei 6 continenti
- Adiacenze simmetriche (se A→B allora B→A)
- 16 obiettivi, ognuno con 20-23 territori esistenti
- Mazzo carte: 44 totali (42 territori + 2 jolly), simboli ben distribuiti
- Soglie sdadata corrette per round e giocatore
- Strutture dati base funzionanti

Esegui: python -m pytest tests/test_data.py -v
       oppure: python tests/test_data.py
"""

import sys
import os

# Aggiungi root al path per poter importare risiko_env
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env.data import (
    CONTINENTI,
    TUTTI_TERRITORI,
    CONTINENTE_DI,
    BONUS_CONTINENTE,
    ADIACENZE,
    OBIETTIVI,
    SIMBOLO_CARTA,
    NUM_JOLLY,
    NUM_GIOCATORI,
    COLORI_GIOCATORI,
    MAX_ARMATE_TOTALI,
    MAX_CARTE_MANO,
    soglia_sdadata,
    confinanti,
    punti_territorio,
    FANTE,
    CANNONE,
    CAVALLO,
    JOLLY,
)
from risiko_env.stato import (
    Carta,
    TerritorioStato,
    Giocatore,
    StatoPartita,
    Fase,
    crea_mazzo_completo,
)


def test_42_territori_totali():
    """Devono esserci esattamente 42 territori."""
    assert len(TUTTI_TERRITORI) == 42, f"Trovati {len(TUTTI_TERRITORI)} territori invece di 42"
    print("✓ 42 territori totali")


def test_continenti_e_dimensioni():
    """Ogni continente deve avere il numero atteso di territori."""
    attesi = {
        "nordamerica": 9,
        "sudamerica": 4,
        "europa": 7,
        "africa": 6,
        "asia": 12,
        "oceania": 4,
    }
    for cont, num in attesi.items():
        actual = len(CONTINENTI[cont])
        assert actual == num, f"{cont}: trovati {actual} territori, attesi {num}"
    print("✓ Dimensioni continenti corrette (NA=9, SA=4, EU=7, AF=6, AS=12, OC=4)")


def test_no_territori_duplicati():
    """Nessun territorio deve apparire in due continenti."""
    visti = set()
    for cont, terrs in CONTINENTI.items():
        for t in terrs:
            assert t not in visti, f"Territorio {t} duplicato!"
            visti.add(t)
    assert len(visti) == 42
    print("✓ Nessun territorio duplicato tra continenti")


def test_continente_di_inverso():
    """La mappa CONTINENTE_DI deve essere inversa coerente di CONTINENTI."""
    for t in TUTTI_TERRITORI:
        cont = CONTINENTE_DI[t]
        assert t in CONTINENTI[cont], f"{t} dovrebbe essere in {cont}"
    print("✓ Mappa CONTINENTE_DI coerente")


def test_bonus_continenti():
    """Verifica i bonus standard RisiKo."""
    assert BONUS_CONTINENTE["nordamerica"] == 5
    assert BONUS_CONTINENTE["sudamerica"] == 2
    assert BONUS_CONTINENTE["europa"] == 5
    assert BONUS_CONTINENTE["africa"] == 3
    assert BONUS_CONTINENTE["asia"] == 7
    assert BONUS_CONTINENTE["oceania"] == 2
    print("✓ Bonus continenti corretti")


def test_adiacenze_simmetriche():
    """
    Se A è adiacente a B, allora B deve essere adiacente ad A.
    Test critico: senza simmetria il combattimento si rompe.
    """
    for t, vicini in ADIACENZE.items():
        for v in vicini:
            assert t in ADIACENZE[v], (
                f"Asimmetria! {t} ha {v} come vicino, "
                f"ma {v} non ha {t} come vicino. "
                f"Vicini di {v}: {sorted(ADIACENZE[v])}"
            )
    print("✓ Adiacenze simmetriche")


def test_adiacenze_coprono_tutti_i_42():
    """Ogni territorio deve avere almeno 1 vicino e tutti i 42 devono essere chiavi."""
    for t in TUTTI_TERRITORI:
        assert t in ADIACENZE, f"Territorio {t} non ha adiacenze definite"
        assert len(ADIACENZE[t]) >= 1, f"Territorio {t} senza vicini"
    print("✓ Tutti i 42 territori hanno adiacenze definite")


def test_alcune_adiacenze_specifiche():
    """Spot-check su adiacenze critiche del RisiKo italiano."""
    # Brasile è l'unico ponte tra Sud America e Africa
    assert "africa_del_nord" in ADIACENZE["brasile"]
    assert "brasile" in ADIACENZE["africa_del_nord"]

    # Kamchatka collega Asia e Nord America
    assert "alaska" in ADIACENZE["kamchatka"]
    assert "kamchatka" in ADIACENZE["alaska"]

    # Madagascar è isolato (solo africa_orientale e africa_del_sud)
    assert ADIACENZE["madagascar"] == frozenset({"africa_orientale", "africa_del_sud"})

    # Argentina ha solo 2 vicini
    assert ADIACENZE["argentina"] == frozenset({"peru", "brasile"})

    print("✓ Adiacenze critiche corrette (Brasile-Africa, Kamchatka-Alaska, Madagascar, Argentina)")


def test_punti_territorio():
    """punti_territorio = numero di adiacenze."""
    # Argentina: 2 vicini → 2 punti
    assert punti_territorio("argentina") == 2
    # Madagascar: 2 vicini → 2 punti
    assert punti_territorio("madagascar") == 2
    # Cina: 6 vicini → 6 punti
    assert punti_territorio("cina") == 6
    # Ucraina: 6 vicini → 6 punti
    assert punti_territorio("ucraina") == 6
    print("✓ Punti territorio = numero adiacenze")


def test_16_obiettivi():
    """Devono esserci esattamente 16 obiettivi numerati 1-16."""
    assert len(OBIETTIVI) == 16
    assert set(OBIETTIVI.keys()) == set(range(1, 17))
    print("✓ 16 obiettivi numerati 1-16")


def test_obiettivi_dimensioni():
    """Ogni obiettivo deve avere tra 19 e 24 territori (nostro range osservato)."""
    for oid, obj in OBIETTIVI.items():
        n = len(obj["territori"])
        assert 19 <= n <= 24, f"Obiettivo {oid} ({obj['nome']}): {n} territori, fuori range 19-24"
    print("✓ Tutti gli obiettivi hanno 19-24 territori")


def test_obiettivi_territori_validi():
    """Tutti i territori menzionati negli obiettivi devono esistere."""
    insieme_territori = set(TUTTI_TERRITORI)
    for oid, obj in OBIETTIVI.items():
        for t in obj["territori"]:
            assert t in insieme_territori, (
                f"Obiettivo {oid} ({obj['nome']}) contiene territorio sconosciuto: {t}"
            )
    print("✓ Tutti i territori degli obiettivi esistono nella mappa")


def test_obiettivi_nomi_unici():
    """Ogni obiettivo deve avere un nome diverso."""
    nomi = [obj["nome"] for obj in OBIETTIVI.values()]
    assert len(set(nomi)) == len(nomi), f"Nomi duplicati: {nomi}"
    print(f"✓ 16 nomi obiettivi unici: {', '.join(nomi)}")


def test_simboli_carte_distribuzione():
    """14 fanti, 14 cannoni, 14 cavalli (distribuzione standard 42/3)."""
    contatore = {FANTE: 0, CANNONE: 0, CAVALLO: 0}
    for t in TUTTI_TERRITORI:
        sim = SIMBOLO_CARTA[t]
        contatore[sim] += 1
    assert contatore[FANTE] == 14, f"Fanti: {contatore[FANTE]}"
    assert contatore[CANNONE] == 14, f"Cannoni: {contatore[CANNONE]}"
    assert contatore[CAVALLO] == 14, f"Cavalli: {contatore[CAVALLO]}"
    print("✓ Simboli carte distribuiti 14-14-14")


def test_mazzo_completo_44_carte():
    """Il mazzo completo deve avere 44 carte: 42 territori + 2 jolly."""
    mazzo = crea_mazzo_completo()
    assert len(mazzo) == 44
    n_jolly = sum(1 for c in mazzo if c.is_jolly)
    n_territorio = sum(1 for c in mazzo if not c.is_jolly)
    assert n_jolly == 2
    assert n_territorio == 42
    print("✓ Mazzo completo: 44 carte (42 territori + 2 jolly)")


def test_soglie_sdadata():
    """Verifica le soglie sdadata per round e giocatore (specifica 7.2.4)."""
    # Giallo
    assert soglia_sdadata("GIALLO", 34) is None  # troppo presto
    assert soglia_sdadata("GIALLO", 35) == 4
    assert soglia_sdadata("GIALLO", 36) == 5
    assert soglia_sdadata("GIALLO", 37) == 6
    assert soglia_sdadata("GIALLO", 38) == 7
    assert soglia_sdadata("GIALLO", 50) == 7  # plateau

    # Blu/Rosso/Verde
    for col in ["BLU", "ROSSO", "VERDE"]:
        assert soglia_sdadata(col, 35) is None  # troppo presto
        assert soglia_sdadata(col, 36) == 4
        assert soglia_sdadata(col, 37) == 5
        assert soglia_sdadata(col, 38) == 6
        assert soglia_sdadata(col, 39) == 7
        assert soglia_sdadata(col, 50) == 7  # plateau

    print("✓ Soglie sdadata corrette per Giallo (35-38) e altri (36-39)")


def test_carta_struttura():
    """Le carte devono avere struttura coerente."""
    c1 = Carta(territorio="brasile", simbolo=FANTE)
    assert not c1.is_jolly
    c2 = Carta(territorio=None, simbolo=JOLLY)
    assert c2.is_jolly
    print("✓ Struttura Carta funziona")


def test_stato_partita_inizializzazione():
    """StatoPartita deve auto-inizializzare mappa e giocatori."""
    stato = StatoPartita()
    # 42 territori inizializzati
    assert len(stato.mappa) == 42
    for t in TUTTI_TERRITORI:
        assert t in stato.mappa
        assert stato.mappa[t].proprietario is None
        assert stato.mappa[t].armate == 0

    # 4 giocatori inizializzati nell'ordine
    assert len(stato.giocatori) == 4
    assert stato.giocatori["BLU"].ordine_mano == 1
    assert stato.giocatori["ROSSO"].ordine_mano == 2
    assert stato.giocatori["VERDE"].ordine_mano == 3
    assert stato.giocatori["GIALLO"].ordine_mano == 4

    # Tutti vivi all'inizio
    assert stato.giocatori_vivi() == ["BLU", "ROSSO", "VERDE", "GIALLO"]

    # Nessun obiettivo assegnato
    for g in stato.giocatori.values():
        assert g.obiettivo_id is None
        assert g.carte == []
        assert g.vivo

    # Partita non iniziata
    assert stato.round_corrente == 0
    assert stato.giocatore_corrente is None
    assert stato.vincitore is None
    assert not stato.terminata

    print("✓ StatoPartita inizializza correttamente 42 territori e 4 giocatori")


def test_giocatore_helpers():
    """Helper di Giocatore funzionanti."""
    g = Giocatore(colore="BLU", ordine_mano=1)
    assert g.num_carte() == 0
    assert g.puo_pescare()

    # Aggiungi 7 carte → non può più pescare
    for i in range(7):
        g.carte.append(Carta(territorio=TUTTI_TERRITORI[i], simbolo=FANTE))
    assert g.num_carte() == 7
    assert not g.puo_pescare()

    print("✓ Helper Giocatore (num_carte, puo_pescare)")


def test_fasi_ordine():
    """Le 4 fasi sono nell'ordine corretto."""
    assert Fase.ORDINE == [
        Fase.TRIS_E_RINFORZI,
        Fase.ATTACCHI,
        Fase.SPOSTAMENTO,
        Fase.PESCA_CARTA,
    ]
    print("✓ Ordine fasi: TRIS_E_RINFORZI → ATTACCHI → SPOSTAMENTO → PESCA_CARTA")


def test_costanti_numeriche():
    """Verifica le costanti numeriche chiave."""
    assert NUM_GIOCATORI == 4
    assert COLORI_GIOCATORI == ["BLU", "ROSSO", "VERDE", "GIALLO"]
    assert MAX_ARMATE_TOTALI == 130
    assert MAX_CARTE_MANO == 7
    assert NUM_JOLLY == 2
    print("✓ Costanti numeriche corrette")


# ─────────────────────────────────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────────────────────────────────

def main():
    """Esegue tutti i test in sequenza."""
    tests = [
        test_42_territori_totali,
        test_continenti_e_dimensioni,
        test_no_territori_duplicati,
        test_continente_di_inverso,
        test_bonus_continenti,
        test_adiacenze_simmetriche,
        test_adiacenze_coprono_tutti_i_42,
        test_alcune_adiacenze_specifiche,
        test_punti_territorio,
        test_16_obiettivi,
        test_obiettivi_dimensioni,
        test_obiettivi_territori_validi,
        test_obiettivi_nomi_unici,
        test_simboli_carte_distribuzione,
        test_mazzo_completo_44_carte,
        test_soglie_sdadata,
        test_carta_struttura,
        test_stato_partita_inizializzazione,
        test_giocatore_helpers,
        test_fasi_ordine,
        test_costanti_numeriche,
    ]

    print("\n" + "=" * 60)
    print("Test Modulo 1: Dati statici e strutture base")
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
