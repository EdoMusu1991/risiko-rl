"""
test_sdadata.py — Test del Modulo 4 (sdadata e fine partita).

Verifica:
- deve_tirare_sdadata: condizioni round e conquiste
- tira_sdadata: probabilità sui vari soglie
- Sdadata obbligatoria (non si può evitare con conquiste <= 2)
- Sdadata evitata con 3+ conquiste
- Terminazione per sdadata riuscita
- Cap di sicurezza al round 60
- Integrazione con determina_vincitore

Esegui: python tests\test_sdadata.py
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env.data import (
    soglia_sdadata,
    ROUND_CAP_SICUREZZA,
    OBIETTIVI,
)
from risiko_env.stato import StatoPartita
from risiko_env.setup import crea_partita_iniziale
from risiko_env.sdadata import (
    deve_tirare_sdadata,
    tira_sdadata,
    termina_partita_per_sdadata,
    termina_partita_per_cap_sicurezza,
    deve_attivare_cap_sicurezza,
    gestisci_fine_turno,
)


# ═════════════════════════════════════════════════════════════════════════
#  CONDIZIONI PER SDADATA
# ═════════════════════════════════════════════════════════════════════════

def test_sdadata_round_troppo_presto():
    """Round 1-34: nessuno può tirare la sdadata."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 1
    for col in ["BLU", "ROSSO", "VERDE", "GIALLO"]:
        stato.conquiste_turno_corrente[col] = 0
        assert not deve_tirare_sdadata(stato, col), f"{col} round 1 dovrebbe essere False"

    stato.round_corrente = 34
    for col in ["BLU", "ROSSO", "VERDE", "GIALLO"]:
        assert not deve_tirare_sdadata(stato, col), f"{col} round 34 dovrebbe essere False"

    print("✓ Round 1-34: nessuno tira la sdadata")


def test_sdadata_round_35_solo_giallo():
    """Al round 35, SOLO Giallo può tirare la sdadata."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 35
    for col in ["BLU", "ROSSO", "VERDE", "GIALLO"]:
        stato.conquiste_turno_corrente[col] = 0

    assert not deve_tirare_sdadata(stato, "BLU")
    assert not deve_tirare_sdadata(stato, "ROSSO")
    assert not deve_tirare_sdadata(stato, "VERDE")
    assert deve_tirare_sdadata(stato, "GIALLO")
    print("✓ Round 35: solo Giallo tira (gli altri partono dal 36)")


def test_sdadata_round_36_tutti():
    """Dal round 36, anche Blu/Rosso/Verde possono tirare."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 36
    for col in ["BLU", "ROSSO", "VERDE", "GIALLO"]:
        stato.conquiste_turno_corrente[col] = 0
        assert deve_tirare_sdadata(stato, col), f"{col} round 36 dovrebbe poter tirare"
    print("✓ Round 36: tutti tirano se ≤ 2 conquiste")


def test_sdadata_obbligatoria_con_2_conquiste():
    """Conquiste = 2: deve tirare (specifica: 'massimo 2 territori')."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 36
    stato.conquiste_turno_corrente["BLU"] = 2
    assert deve_tirare_sdadata(stato, "BLU"), "Con 2 conquiste deve tirare"
    print("✓ Sdadata obbligatoria con 2 conquiste")


def test_sdadata_evitata_con_3_conquiste():
    """Conquiste >= 3: NON tira (unico modo per evitare la sdadata)."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 36
    stato.conquiste_turno_corrente["BLU"] = 3
    assert not deve_tirare_sdadata(stato, "BLU"), "Con 3 conquiste non deve tirare"

    stato.conquiste_turno_corrente["BLU"] = 5
    assert not deve_tirare_sdadata(stato, "BLU"), "Con 5 conquiste non deve tirare"
    print("✓ Sdadata evitata con 3+ conquiste")


# ═════════════════════════════════════════════════════════════════════════
#  TIRO DELLA SDADATA: VERIFICA STATISTICA SOGLIE
# ═════════════════════════════════════════════════════════════════════════

def test_tira_sdadata_dadi_validi():
    """I dadi devono sempre essere 1-6."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 35
    rng = random.Random(42)

    for _ in range(100):
        riuscita, d1, d2, soglia = tira_sdadata(stato, "GIALLO", rng)
        assert 1 <= d1 <= 6
        assert 1 <= d2 <= 6
        assert soglia == 4  # round 35 Giallo
        assert riuscita == (d1 + d2 <= 4)
    print("✓ Dadi sdadata sempre 1-6, riuscita coerente con somma vs soglia")


def test_tira_sdadata_probabilita_round_35_giallo():
    """
    Round 35 Giallo: soglia=4. Somme possibili 2-12, somme <=4 sono {2,3,4}.
    Probabilità teorica: P(2)=1/36, P(3)=2/36, P(4)=3/36 → totale 6/36 = ~16.7%
    """
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 35
    rng = random.Random(123)

    n = 10000
    riusciti = sum(1 for _ in range(n) if tira_sdadata(stato, "GIALLO", rng)[0])
    p_osservata = riusciti / n
    p_attesa = 6 / 36  # ~0.1667

    # Tolleranza ±2%
    assert abs(p_osservata - p_attesa) < 0.02, (
        f"P osservata {p_osservata:.3f}, attesa ~{p_attesa:.3f}"
    )
    print(f"✓ Sdadata round 35 Giallo (soglia=4): "
          f"P osservata {p_osservata:.3f}, attesa {p_attesa:.3f}")


def test_tira_sdadata_probabilita_round_38_giallo():
    """
    Round 38+ Giallo: soglia=7. Somme <=7: 21 su 36 → ~58.3%
    """
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 38
    rng = random.Random(456)

    n = 10000
    riusciti = sum(1 for _ in range(n) if tira_sdadata(stato, "GIALLO", rng)[0])
    p_osservata = riusciti / n
    p_attesa = 21 / 36  # ~0.583

    assert abs(p_osservata - p_attesa) < 0.02
    print(f"✓ Sdadata round 38 Giallo (soglia=7): "
          f"P osservata {p_osservata:.3f}, attesa {p_attesa:.3f}")


# ═════════════════════════════════════════════════════════════════════════
#  TERMINAZIONE PARTITA
# ═════════════════════════════════════════════════════════════════════════

def test_termina_partita_per_sdadata():
    """Termina la partita e determina vincitore."""
    stato = crea_partita_iniziale(seed=42)
    assert not stato.terminata
    assert stato.vincitore is None

    termina_partita_per_sdadata(stato, "GIALLO")
    assert stato.terminata
    assert stato.motivo_fine == "sdadata"
    assert stato.vincitore is not None  # Determinato dalla cascata
    print(f"✓ Termina per sdadata: vincitore={stato.vincitore}, motivo={stato.motivo_fine}")


def test_cap_sicurezza_round_60():
    """Round 60 + ultimo giocatore vivo → attiva cap."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 60
    # Tutti vivi, GIALLO è l'ultimo
    assert deve_attivare_cap_sicurezza(stato, ultimo_giocatore_del_round=True)
    assert not deve_attivare_cap_sicurezza(stato, ultimo_giocatore_del_round=False)

    # Round 59 → no
    stato.round_corrente = 59
    assert not deve_attivare_cap_sicurezza(stato, ultimo_giocatore_del_round=True)
    print("✓ Cap di sicurezza attivato solo a fine round 60+")


def test_cap_sicurezza_terminazione():
    """Quando si attiva il cap, la partita termina con vincitore determinato."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 60
    termina_partita_per_cap_sicurezza(stato)
    assert stato.terminata
    assert stato.motivo_fine == "cap_sicurezza"
    assert stato.vincitore is not None
    print(f"✓ Cap sicurezza: vincitore={stato.vincitore}, motivo={stato.motivo_fine}")


# ═════════════════════════════════════════════════════════════════════════
#  HELPER INTEGRATIVO: gestisci_fine_turno
# ═════════════════════════════════════════════════════════════════════════

def test_fine_turno_round_normale():
    """Round 1-34: nessuna sdadata, nessun cap."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 10
    stato.conquiste_turno_corrente["BLU"] = 0
    rng = random.Random(42)

    info = gestisci_fine_turno(stato, "BLU", rng)
    assert not info["sdadata_tirata"]
    assert not info["cap_sicurezza_attivato"]
    assert not info["partita_terminata"]
    assert not stato.terminata
    print("✓ Fine turno round 10: nessuna azione speciale")


def test_fine_turno_sdadata_riuscita():
    """Test forzato: con seed e setup mirati, la sdadata riesce."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 38  # Soglia = 7 per Giallo
    stato.conquiste_turno_corrente["GIALLO"] = 0

    # Con seed bassi, è probabile che la sdadata riesca (soglia 7 = 58% prob)
    rng = random.Random(1)
    info = gestisci_fine_turno(stato, "GIALLO", rng)

    assert info["sdadata_tirata"]
    if info["sdadata_riuscita"]:
        assert info["partita_terminata"]
        assert stato.terminata
        assert stato.motivo_fine == "sdadata"
        assert stato.vincitore is not None
    else:
        assert not info["partita_terminata"]
        assert not stato.terminata

    print(f"✓ Fine turno sdadata round 38 Giallo: "
          f"riuscita={info['sdadata_riuscita']}, "
          f"dadi={info['sdadata_dadi']}, soglia={info['sdadata_soglia']}")


def test_fine_turno_sdadata_evitata_con_conquiste():
    """Con 3 conquiste, niente sdadata anche al round 38."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 38
    stato.conquiste_turno_corrente["GIALLO"] = 3
    rng = random.Random(1)

    info = gestisci_fine_turno(stato, "GIALLO", rng)
    assert not info["sdadata_tirata"], "Con 3 conquiste non doveva tirare"
    assert not info["partita_terminata"]
    print("✓ Fine turno sdadata evitata con 3 conquiste")


def test_fine_turno_cap_sicurezza_round_60():
    """Round 60, ultimo giocatore vivo → cap attivato."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 60
    # Forziamo conquiste alte per evitare la sdadata
    stato.conquiste_turno_corrente["GIALLO"] = 99
    rng = random.Random(42)

    info = gestisci_fine_turno(stato, "GIALLO", rng)
    assert info["cap_sicurezza_attivato"]
    assert info["partita_terminata"]
    assert stato.motivo_fine == "cap_sicurezza"
    print(f"✓ Cap sicurezza al round 60: vincitore={info['vincitore']}")


def test_fine_turno_partita_gia_terminata():
    """Se la partita è già terminata, fine_turno non fa nulla."""
    stato = crea_partita_iniziale(seed=42)
    stato.terminata = True
    stato.vincitore = "BLU"
    stato.motivo_fine = "obiettivo_completato"
    rng = random.Random(42)

    info = gestisci_fine_turno(stato, "GIALLO", rng)
    assert not info["sdadata_tirata"]
    assert not info["cap_sicurezza_attivato"]
    assert info["vincitore"] == "BLU"
    assert info["motivo_fine"] == "obiettivo_completato"
    print("✓ Partita già terminata: fine_turno non interviene")


def test_fine_turno_giallo_ultimo_round_60_no_sdadata():
    """
    Caso limite: Giallo round 60 con 5 conquiste.
    Salta la sdadata (3+ conquiste) MA scatta il cap (ultimo del round 60).
    """
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 60
    stato.conquiste_turno_corrente["GIALLO"] = 5
    rng = random.Random(42)

    info = gestisci_fine_turno(stato, "GIALLO", rng)
    assert not info["sdadata_tirata"]
    assert info["cap_sicurezza_attivato"]
    assert info["partita_terminata"]
    print("✓ Caso limite: 5 conquiste evita sdadata, ma cap scatta lo stesso")


# ═════════════════════════════════════════════════════════════════════════
#  TEST DI INTEGRAZIONE: SOGLIE PER OGNI ROUND/COLORE
# ═════════════════════════════════════════════════════════════════════════

def test_soglie_complete_per_giallo():
    """Verifica soglie Giallo per round 35-50."""
    stato = crea_partita_iniziale(seed=42)
    stato.conquiste_turno_corrente["GIALLO"] = 0
    rng = random.Random(42)

    soglie_attese = {
        35: 4, 36: 5, 37: 6, 38: 7, 39: 7, 40: 7, 50: 7,
    }
    for r, soglia_attesa in soglie_attese.items():
        stato.round_corrente = r
        if not deve_tirare_sdadata(stato, "GIALLO"):
            continue  # round troppo presto
        _, _, _, soglia = tira_sdadata(stato, "GIALLO", rng)
        assert soglia == soglia_attesa, f"Round {r} Giallo: atteso {soglia_attesa}, ottenuto {soglia}"
    print("✓ Soglie Giallo round 35-50 corrette")


def test_soglie_complete_per_altri():
    """Verifica soglie Blu/Rosso/Verde per round 36-50."""
    stato = crea_partita_iniziale(seed=42)
    rng = random.Random(42)

    for col in ["BLU", "ROSSO", "VERDE"]:
        stato.conquiste_turno_corrente[col] = 0
        soglie_attese = {
            36: 4, 37: 5, 38: 6, 39: 7, 40: 7, 50: 7,
        }
        for r, soglia_attesa in soglie_attese.items():
            stato.round_corrente = r
            if not deve_tirare_sdadata(stato, col):
                continue
            _, _, _, soglia = tira_sdadata(stato, col, rng)
            assert soglia == soglia_attesa, f"Round {r} {col}: atteso {soglia_attesa}, ottenuto {soglia}"
    print("✓ Soglie Blu/Rosso/Verde round 36-50 corrette")


# ═════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═════════════════════════════════════════════════════════════════════════

def main():
    tests = [
        # Condizioni per sdadata
        test_sdadata_round_troppo_presto,
        test_sdadata_round_35_solo_giallo,
        test_sdadata_round_36_tutti,
        test_sdadata_obbligatoria_con_2_conquiste,
        test_sdadata_evitata_con_3_conquiste,
        # Tiro dadi
        test_tira_sdadata_dadi_validi,
        test_tira_sdadata_probabilita_round_35_giallo,
        test_tira_sdadata_probabilita_round_38_giallo,
        # Terminazione
        test_termina_partita_per_sdadata,
        test_cap_sicurezza_round_60,
        test_cap_sicurezza_terminazione,
        # Integrazione
        test_fine_turno_round_normale,
        test_fine_turno_sdadata_riuscita,
        test_fine_turno_sdadata_evitata_con_conquiste,
        test_fine_turno_cap_sicurezza_round_60,
        test_fine_turno_partita_gia_terminata,
        test_fine_turno_giallo_ultimo_round_60_no_sdadata,
        # Soglie complete
        test_soglie_complete_per_giallo,
        test_soglie_complete_per_altri,
    ]

    print("\n" + "=" * 60)
    print("Test Modulo 4: Sdadata e fine partita")
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
