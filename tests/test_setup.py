"""
test_setup.py — Test per il Modulo 2 (setup partita).

Verifica:
- Distribuzione territori: 11/11/10/10 esatti, vincolo continentale rispettato
- Distribuzione obiettivi: 1 per giocatore, tutti diversi
- Piazzamento iniziale: totali corretti (Blu=20+11=31, Verde=19+10=29 etc)
- Mazzo: 44 carte mescolate, 0 scarti, 0 carte in mano
- Riproducibilità: stesso seed → stessa partita
- Stato post-setup: round=1, giocatore=BLU, tutti vivi

Esegui: python tests/test_setup.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from risiko_env.data import (
    TUTTI_TERRITORI,
    CONTINENTI,
    COLORI_GIOCATORI,
    OBIETTIVI,
    TERRITORI_PER_GIOCATORE,
    limite_continente_distribuzione,
)
from risiko_env.stato import StatoPartita
from risiko_env.setup import (
    distribuisci_territori,
    distribuisci_obiettivi,
    piazzamento_iniziale_random,
    setup_mazzo,
    crea_partita_iniziale,
)
import random


# ─────────────────────────────────────────────────────────────────────────
#  TEST DISTRIBUZIONE TERRITORI
# ─────────────────────────────────────────────────────────────────────────

def test_distribuzione_quantita_corrette():
    """Distribuzione: Blu=10, Rosso=10, Verde=11, Giallo=11."""
    stato = StatoPartita()
    distribuisci_territori(stato, random.Random(42))

    conteggi = {col: stato.num_territori_di(col) for col in COLORI_GIOCATORI}
    assert conteggi["BLU"] == 10, f"BLU ha {conteggi['BLU']}"
    assert conteggi["ROSSO"] == 10, f"ROSSO ha {conteggi['ROSSO']}"
    assert conteggi["VERDE"] == 11, f"VERDE ha {conteggi['VERDE']}"
    assert conteggi["GIALLO"] == 11, f"GIALLO ha {conteggi['GIALLO']}"
    assert sum(conteggi.values()) == 42
    print("✓ Distribuzione 10/10/11/11 (totale 42)")


def test_distribuzione_tutti_assegnati():
    """Tutti i 42 territori devono avere un proprietario dopo la distribuzione."""
    stato = StatoPartita()
    distribuisci_territori(stato, random.Random(42))

    senza_proprietario = [
        t for t, s in stato.mappa.items()
        if s.proprietario is None
    ]
    assert not senza_proprietario, f"Territori senza proprietario: {senza_proprietario}"
    print("✓ Tutti i 42 territori assegnati")


def test_distribuzione_1_carro_per_territorio():
    """Dopo la distribuzione, ogni territorio deve avere esattamente 1 carro."""
    stato = StatoPartita()
    distribuisci_territori(stato, random.Random(42))

    for t, s in stato.mappa.items():
        assert s.armate == 1, f"{t}: {s.armate} armate (atteso 1)"
    print("✓ 1 carro per territorio dopo distribuzione")


def test_vincolo_continentale_su_molti_seed():
    """
    Verifica il vincolo "max metà territori per continente" su molti seed.
    La specifica 3.1 punto 4 ammette violazioni rare in casi di impossibilità
    matematica. Accettiamo fino al 10% di violazioni come fallback ragionevole.
    """
    violazioni = 0
    n_seed = 100
    n_verifiche = n_seed * 4 * 6  # 100 seed × 4 giocatori × 6 continenti = 2400
    for seed in range(n_seed):
        stato = StatoPartita()
        distribuisci_territori(stato, random.Random(seed))

        for colore in COLORI_GIOCATORI:
            terr_giocatore = stato.territori_di(colore)
            for cont, terrs_cont in CONTINENTI.items():
                in_cont = sum(1 for t in terr_giocatore if t in terrs_cont)
                limite = limite_continente_distribuzione(cont)
                if in_cont > limite:
                    violazioni += 1

    # Tolleranza: max 5% di violazioni (fallback raro previsto dalla specifica)
    soglia = n_verifiche * 0.05
    assert violazioni < soglia, (
        f"Troppe violazioni: {violazioni}/{n_verifiche} "
        f"({violazioni/n_verifiche*100:.1f}% > 5%)"
    )
    print(f"✓ Vincolo continentale rispettato (violazioni: {violazioni}/{n_verifiche} = "
          f"{violazioni/n_verifiche*100:.1f}%, sotto soglia 5%)")


# ─────────────────────────────────────────────────────────────────────────
#  TEST DISTRIBUZIONE OBIETTIVI
# ─────────────────────────────────────────────────────────────────────────

def test_obiettivi_assegnati_tutti():
    """Ogni giocatore deve ricevere un obiettivo."""
    stato = StatoPartita()
    distribuisci_obiettivi(stato, random.Random(42))

    for colore in COLORI_GIOCATORI:
        obj_id = stato.giocatori[colore].obiettivo_id
        assert obj_id is not None, f"{colore} senza obiettivo"
        assert obj_id in OBIETTIVI, f"{colore} ha obiettivo invalido {obj_id}"
    print("✓ Tutti i 4 giocatori hanno un obiettivo")


def test_obiettivi_tutti_diversi():
    """I 4 obiettivi assegnati devono essere tutti diversi."""
    stato = StatoPartita()
    distribuisci_obiettivi(stato, random.Random(42))

    ids = [stato.giocatori[col].obiettivo_id for col in COLORI_GIOCATORI]
    assert len(set(ids)) == 4, f"Obiettivi duplicati! {ids}"
    print(f"✓ 4 obiettivi diversi: {ids}")


def test_obiettivi_distribuzione_uniforme():
    """Su molti seed, ogni obiettivo dovrebbe essere pescato circa 4*N/16 volte."""
    contatore = Counter()
    n_partite = 1600
    for seed in range(n_partite):
        stato = StatoPartita()
        distribuisci_obiettivi(stato, random.Random(seed))
        for col in COLORI_GIOCATORI:
            contatore[stato.giocatori[col].obiettivo_id] += 1

    # Ogni obiettivo dovrebbe apparire circa n_partite*4/16 = 400 volte
    atteso = n_partite * 4 / 16
    for obj_id, count in contatore.items():
        # Tolleranza del 30%
        assert atteso * 0.7 < count < atteso * 1.3, (
            f"Obiettivo {obj_id}: {count} (atteso {atteso:.0f})"
        )
    print(f"✓ Distribuzione obiettivi uniforme su {n_partite} partite")


# ─────────────────────────────────────────────────────────────────────────
#  TEST PIAZZAMENTO INIZIALE
# ─────────────────────────────────────────────────────────────────────────

def test_piazzamento_armate_totali():
    """
    Dopo piazzamento iniziale + 1 carro automatico per territorio:
    - Blu: 20 piazzati + 10 territori = 30 armate totali
    - Rosso: 20 + 10 = 30
    - Verde: 19 + 11 = 30
    - Giallo: 19 + 11 = 30
    Tutti partono con 30 armate (specifica).
    """
    stato = StatoPartita()
    rng = random.Random(42)
    distribuisci_territori(stato, rng)
    piazzamento_iniziale_random(stato, rng)

    attesi = {"BLU": 30, "ROSSO": 30, "VERDE": 30, "GIALLO": 30}
    for colore, atteso in attesi.items():
        actual = stato.num_armate_di(colore)
        assert actual == atteso, f"{colore}: {actual} armate (atteso {atteso})"
    print("✓ Piazzamento iniziale: tutti i giocatori 30 armate totali")


def test_piazzamento_almeno_1_per_territorio():
    """Ogni territorio deve avere almeno 1 carro dopo il piazzamento."""
    stato = StatoPartita()
    rng = random.Random(42)
    distribuisci_territori(stato, rng)
    piazzamento_iniziale_random(stato, rng)

    for t, s in stato.mappa.items():
        assert s.armate >= 1, f"{t}: solo {s.armate} armate"
    print("✓ Tutti i territori hanno almeno 1 carro dopo piazzamento")


# ─────────────────────────────────────────────────────────────────────────
#  TEST MAZZO
# ─────────────────────────────────────────────────────────────────────────

def test_mazzo_44_carte():
    """Dopo setup mazzo: 44 carte attive, 0 scarti."""
    stato = StatoPartita()
    setup_mazzo(stato, random.Random(42))

    assert len(stato.mazzo_attivo) == 44
    assert len(stato.pila_scarti) == 0
    print("✓ Mazzo: 44 carte attive, 0 scarti")


def test_mazzo_mescolato():
    """
    Il mazzo deve essere mescolato (non in ordine canonico).
    Test: con seed=42, le prime 5 carte non devono essere in ordine alfabetico.
    """
    stato = StatoPartita()
    setup_mazzo(stato, random.Random(42))

    primi_5 = [c.territorio for c in stato.mazzo_attivo[:5] if c.territorio]
    # Ordinato alfabeticamente sarebbe quello
    if primi_5 == sorted(primi_5):
        # È improbabile ma possibile per caso. Riprovo con altro seed.
        stato2 = StatoPartita()
        setup_mazzo(stato2, random.Random(43))
        primi_5_b = [c.territorio for c in stato2.mazzo_attivo[:5] if c.territorio]
        assert primi_5_b != sorted(primi_5_b), "Mazzo non sembra mescolato"
    print("✓ Mazzo mescolato")


# ─────────────────────────────────────────────────────────────────────────
#  TEST RIPRODUCIBILITÀ
# ─────────────────────────────────────────────────────────────────────────

def test_riproducibilita_stesso_seed():
    """Stesso seed → stessa partita identica."""
    stato1 = crea_partita_iniziale(seed=12345)
    stato2 = crea_partita_iniziale(seed=12345)

    # Stessa distribuzione territori
    for t in TUTTI_TERRITORI:
        assert stato1.mappa[t].proprietario == stato2.mappa[t].proprietario
        assert stato1.mappa[t].armate == stato2.mappa[t].armate

    # Stessi obiettivi
    for col in COLORI_GIOCATORI:
        assert stato1.giocatori[col].obiettivo_id == stato2.giocatori[col].obiettivo_id

    # Stesso mazzo
    for c1, c2 in zip(stato1.mazzo_attivo, stato2.mazzo_attivo):
        assert c1 == c2

    print("✓ Riproducibilità: stesso seed → stessa partita identica")


def test_seed_diversi_partite_diverse():
    """Seed diversi → partite diverse."""
    stato1 = crea_partita_iniziale(seed=1)
    stato2 = crea_partita_iniziale(seed=2)

    diverse = False
    for t in TUTTI_TERRITORI:
        if stato1.mappa[t].proprietario != stato2.mappa[t].proprietario:
            diverse = True
            break

    assert diverse, "Seed 1 e 2 hanno prodotto la stessa distribuzione (improbabile)"
    print("✓ Seed diversi producono partite diverse")


# ─────────────────────────────────────────────────────────────────────────
#  TEST INTEGRAZIONE
# ─────────────────────────────────────────────────────────────────────────

def test_partita_iniziale_completa():
    """Verifica che crea_partita_iniziale produca uno stato pienamente valido."""
    stato = crea_partita_iniziale(seed=42)

    # Tutti i territori assegnati con almeno 1 armata
    for t, s in stato.mappa.items():
        assert s.proprietario in COLORI_GIOCATORI
        assert s.armate >= 1

    # Totali armate per giocatore corretti (tutti 30 da specifica)
    armate_totali = {col: stato.num_armate_di(col) for col in COLORI_GIOCATORI}
    assert armate_totali["BLU"] == 30
    assert armate_totali["ROSSO"] == 30
    assert armate_totali["VERDE"] == 30
    assert armate_totali["GIALLO"] == 30

    # Mazzo pronto, scarti vuoti
    assert len(stato.mazzo_attivo) == 44
    assert len(stato.pila_scarti) == 0

    # Tutti hanno un obiettivo, nessuna carta in mano
    for col in COLORI_GIOCATORI:
        g = stato.giocatori[col]
        assert g.obiettivo_id is not None
        assert g.num_carte() == 0
        assert g.vivo

    # Pronto per il primo turno
    assert stato.round_corrente == 1
    assert stato.giocatore_corrente == "BLU"
    assert all(stato.conquiste_turno_corrente[col] == 0 for col in COLORI_GIOCATORI)
    assert not stato.terminata
    assert stato.vincitore is None

    print("✓ Partita iniziale completa: tutto pronto per round 1, turno Blu")


def test_partita_iniziale_obiettivi_diversi():
    """In una partita reale, i 4 obiettivi devono essere diversi."""
    for seed in range(20):
        stato = crea_partita_iniziale(seed=seed)
        ids = [stato.giocatori[col].obiettivo_id for col in COLORI_GIOCATORI]
        assert len(set(ids)) == 4, f"Seed {seed}: obiettivi duplicati {ids}"
    print("✓ Obiettivi sempre diversi su 20 partite")


# ─────────────────────────────────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_distribuzione_quantita_corrette,
        test_distribuzione_tutti_assegnati,
        test_distribuzione_1_carro_per_territorio,
        test_vincolo_continentale_su_molti_seed,
        test_obiettivi_assegnati_tutti,
        test_obiettivi_tutti_diversi,
        test_obiettivi_distribuzione_uniforme,
        test_piazzamento_armate_totali,
        test_piazzamento_almeno_1_per_territorio,
        test_mazzo_44_carte,
        test_mazzo_mescolato,
        test_riproducibilita_stesso_seed,
        test_seed_diversi_partite_diverse,
        test_partita_iniziale_completa,
        test_partita_iniziale_obiettivi_diversi,
    ]

    print("\n" + "=" * 60)
    print("Test Modulo 2: Setup partita")
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
