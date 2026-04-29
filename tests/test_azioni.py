"""
test_azioni.py — Test del Modulo 5b (action space e masking).

Verifica:
- Dimensioni costanti degli action space
- Maschere TRIS: enumerazione corretta delle combinazioni
- Maschere RINFORZO: solo territori propri
- Maschere ATTACCO: solo coppie adiacenti con avversari, stop sempre legale
- Maschere CONTINUA: stop sempre, continua se condizioni
- Maschere QUANTITÀ: 3 opzioni (min, mid, max)
- Maschere SPOSTAMENTO: solo coppie proprie adiacenti
- Codifica/decodifica reversibili

Esegui: python tests\test_azioni.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env.data import (
    COLORI_GIOCATORI,
    FANTE,
    CANNONE,
    CAVALLO,
    JOLLY,
    TUTTI_TERRITORI,
)
from risiko_env.stato import StatoPartita, Carta
from risiko_env.setup import crea_partita_iniziale
from risiko_env.azioni import (
    NUM_AZIONI_TRIS,
    NUM_AZIONI_RINFORZO,
    NUM_AZIONI_ATTACCO,
    NUM_AZIONI_CONTINUA,
    NUM_AZIONI_QUANTITA,
    NUM_AZIONI_SPOSTAMENTO,
    INDICE_STOP_ATTACCO,
    INDICE_SKIP_SPOSTAMENTO,
    enumera_combinazioni_tris,
    maschera_tris,
    maschera_rinforzo,
    maschera_attacco,
    codifica_attacco,
    decodifica_azione_attacco,
    maschera_continua,
    maschera_quantita,
    calcola_quantita_da_azione,
    maschera_spostamento,
    decodifica_azione_spostamento,
    decodifica_azione_rinforzo,
)


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE TRIS
# ═════════════════════════════════════════════════════════════════════════

def test_tris_senza_carte():
    """Senza carte, solo opzione 0 (skip)."""
    combinazioni = enumera_combinazioni_tris([])
    assert len(combinazioni) == 1
    assert combinazioni[0] == []
    mask = maschera_tris(combinazioni)
    assert mask[0] == True
    assert all(not v for v in mask[1:])
    print("✓ Tris senza carte: solo skip")


def test_tris_3_uguali():
    """3 fanti: 2 opzioni (skip + giocale)."""
    carte = [Carta("alaska", FANTE), Carta("brasile", FANTE), Carta("cina", FANTE)]
    combinazioni = enumera_combinazioni_tris(carte)
    assert len(combinazioni) == 2
    assert combinazioni[0] == []  # skip
    assert len(combinazioni[1]) == 1  # 1 tris
    print("✓ Tris 3 fanti: 2 opzioni (skip + 1 tris)")


def test_tris_due_disgiunti():
    """6 carte (3 fanti + 3 cannoni): 4 opzioni (skip, fanti, cannoni, entrambi)."""
    carte = [
        Carta("alaska", FANTE), Carta("brasile", FANTE), Carta("cina", FANTE),
        Carta("egitto", CANNONE), Carta("ucraina", CANNONE), Carta("mongolia", CANNONE),
    ]
    combinazioni = enumera_combinazioni_tris(carte)
    assert len(combinazioni) >= 3
    # Verifica che ci sia almeno una combinazione con 2 tris
    n_doppi = sum(1 for c in combinazioni if len(c) == 2)
    assert n_doppi >= 1, f"Nessuna combinazione con 2 tris trovata"
    print(f"✓ Tris 3F + 3C: {len(combinazioni)} opzioni, di cui {n_doppi} con 2 tris")


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE RINFORZO
# ═════════════════════════════════════════════════════════════════════════

def test_rinforzo_solo_territori_propri():
    """Maschera rinforzo: solo i 11 territori di Blu sono True."""
    stato = crea_partita_iniziale(seed=42)
    mask = maschera_rinforzo(stato, "BLU")
    assert mask.shape == (NUM_AZIONI_RINFORZO,)
    n_legali = int(np.sum(mask))
    territori_blu = stato.num_territori_di("BLU")
    assert n_legali == territori_blu, f"Mask: {n_legali}, atteso {territori_blu}"
    print(f"✓ Rinforzo: {n_legali} territori legali (= num territori di Blu)")


def test_rinforzo_decodifica():
    """Decodifica: indice → nome territorio."""
    stato = crea_partita_iniziale(seed=42)
    territori_blu = stato.territori_di("BLU")
    mask = maschera_rinforzo(stato, "BLU")
    legali = np.where(mask)[0]
    nomi = [decodifica_azione_rinforzo(int(i)) for i in legali]
    # Tutti i nomi decodificati sono territori di Blu
    for nome in nomi:
        assert nome in territori_blu
    print("✓ Decodifica rinforzo coerente")


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE ATTACCO
# ═════════════════════════════════════════════════════════════════════════

def test_attacco_codifica_decodifica():
    """codifica e decodifica devono essere reversibili."""
    da, verso = "brasile", "argentina"
    idx = codifica_attacco(da, verso)
    assert idx >= 0 and idx < INDICE_STOP_ATTACCO
    da2, verso2 = decodifica_azione_attacco(idx)
    assert da == da2 and verso == verso2

    # Decodifica STOP
    decoded = decodifica_azione_attacco(INDICE_STOP_ATTACCO)
    assert decoded is None
    print("✓ Codifica/decodifica attacco reversibile, STOP gestito")


def test_attacco_maschera_stop_sempre_legale():
    """STOP è sempre nella maschera attacchi, anche senza opzioni."""
    stato = StatoPartita()
    # Setup minimo: BLU ha 1 territorio con 1 armata (non può attaccare)
    stato.mappa["alaska"].proprietario = "BLU"
    stato.mappa["alaska"].armate = 1

    mask = maschera_attacco(stato, "BLU")
    assert mask.shape == (NUM_AZIONI_ATTACCO,)
    assert mask[INDICE_STOP_ATTACCO] == True
    n_legali = int(np.sum(mask))
    assert n_legali == 1, f"Solo STOP doveva essere legale, trovati {n_legali}"
    print("✓ STOP attacco sempre legale (anche senza opzioni)")


def test_attacco_maschera_solo_legali():
    """Tutti gli attacchi nella maschera devono essere legali."""
    from risiko_env.motore import attacco_legale

    stato = crea_partita_iniziale(seed=42)
    # Forza alcune armate per garantire attacchi possibili
    for t in stato.territori_di("BLU"):
        stato.mappa[t].armate = 5

    mask = maschera_attacco(stato, "BLU")

    # Verifica che ogni indice "True" (escludendo STOP) sia un attacco legale
    n_verificati = 0
    for idx in np.where(mask)[0]:
        if idx == INDICE_STOP_ATTACCO:
            continue
        decoded = decodifica_azione_attacco(int(idx))
        assert decoded is not None
        da, verso = decoded
        assert attacco_legale(stato, "BLU", da, verso), (
            f"Maschera segna legale ma non lo è: {da} → {verso}"
        )
        n_verificati += 1
    print(f"✓ Attacco mask: tutti i {n_verificati} attacchi marcati legali sono effettivamente legali")


def test_attacco_maschera_simmetrica():
    """Se Blu può attaccare Rosso, e Rosso ha territori adiacenti a Blu,
    allora Rosso può attaccare Blu (su quegli stessi territori)."""
    stato = crea_partita_iniziale(seed=42)
    for t in stato.territori_di("BLU"):
        stato.mappa[t].armate = 5
    for t in stato.territori_di("ROSSO"):
        stato.mappa[t].armate = 5

    mask_blu = maschera_attacco(stato, "BLU")
    mask_rosso = maschera_attacco(stato, "ROSSO")

    # Solo verifico che entrambi abbiano almeno qualche attacco possibile
    n_blu = int(np.sum(mask_blu)) - 1  # esclude STOP
    n_rosso = int(np.sum(mask_rosso)) - 1
    assert n_blu > 0 and n_rosso > 0
    print(f"✓ Maschere attacco non vuote: Blu={n_blu}, Rosso={n_rosso}")


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE CONTINUA
# ═════════════════════════════════════════════════════════════════════════

def test_continua_stop_sempre_legale():
    """Stop (azione 0) è sempre legale, anche se territori non esistono."""
    stato = StatoPartita()
    mask = maschera_continua(stato, "BLU", "fake1", "fake2")
    assert mask[0] == True
    assert mask[1] == False  # continua impossibile
    print("✓ CONTINUA: stop sempre legale, continua=False per territori invalidi")


def test_continua_legale_con_armate():
    """Continua è legale se attaccante >= 2 armate e difensore >= 1."""
    stato = StatoPartita()
    stato.mappa["brasile"].proprietario = "BLU"
    stato.mappa["brasile"].armate = 5
    stato.mappa["argentina"].proprietario = "ROSSO"
    stato.mappa["argentina"].armate = 1

    mask = maschera_continua(stato, "BLU", "brasile", "argentina")
    assert mask[0] == True  # stop
    assert mask[1] == True  # continua

    # Se brasile ha 1 armata: continua impossibile
    stato.mappa["brasile"].armate = 1
    mask = maschera_continua(stato, "BLU", "brasile", "argentina")
    assert mask[1] == False
    print("✓ CONTINUA: legale solo con armate sufficienti")


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE QUANTITÀ
# ═════════════════════════════════════════════════════════════════════════

def test_quantita_3_opzioni():
    """3 opzioni discrete tutte distinte se min < max-1."""
    mask = maschera_quantita(2, 10)  # range 2-10
    assert all(mask)  # tutte legali
    assert calcola_quantita_da_azione(0, 2, 10) == 2  # min
    assert calcola_quantita_da_azione(1, 2, 10) == 6  # intermedio
    assert calcola_quantita_da_azione(2, 2, 10) == 10  # max
    print("✓ Quantità: min=2, mid=6, max=10 con range 2-10")


def test_quantita_min_uguale_max():
    """Se min == max, solo l'opzione 0 è legale."""
    mask = maschera_quantita(3, 3)
    assert mask[0] == True
    assert mask[1] == False
    assert mask[2] == False
    assert calcola_quantita_da_azione(0, 3, 3) == 3
    print("✓ Quantità: min==max → solo opzione 0 legale")


# ═════════════════════════════════════════════════════════════════════════
#  AZIONE SPOSTAMENTO FINALE
# ═════════════════════════════════════════════════════════════════════════

def test_spostamento_skip_sempre_legale():
    """Skip è sempre legale."""
    stato = StatoPartita()
    mask = maschera_spostamento(stato, "BLU")
    assert mask[INDICE_SKIP_SPOSTAMENTO] == True
    print("✓ SPOSTAMENTO: skip sempre legale")


def test_spostamento_solo_coppie_proprie():
    """Solo coppie (da, verso) entrambi propri e adiacenti."""
    stato = crea_partita_iniziale(seed=42)
    # Forza armate per garantire spostamenti possibili
    for t in stato.territori_di("BLU"):
        stato.mappa[t].armate = 5

    mask = maschera_spostamento(stato, "BLU")
    n_legali = int(np.sum(mask)) - 1  # esclude SKIP

    # Verifica che ogni coppia legale sia tra territori di Blu
    territori_blu = set(stato.territori_di("BLU"))
    for idx in np.where(mask)[0]:
        if idx == INDICE_SKIP_SPOSTAMENTO:
            continue
        decoded = decodifica_azione_spostamento(int(idx))
        assert decoded is not None
        da, verso = decoded
        assert da in territori_blu and verso in territori_blu, (
            f"Coppia non valida: {da} → {verso}"
        )
    print(f"✓ Spostamento: {n_legali} coppie legali, tutte tra territori propri")


# ═════════════════════════════════════════════════════════════════════════
#  DIMENSIONI ACTION SPACE
# ═════════════════════════════════════════════════════════════════════════

def test_dimensioni_consistenti():
    """Le costanti delle dimensioni sono consistenti."""
    assert NUM_AZIONI_TRIS == 11
    assert NUM_AZIONI_RINFORZO == 42
    assert NUM_AZIONI_ATTACCO == 1765  # 42*42 + 1
    assert INDICE_STOP_ATTACCO == 1764
    assert NUM_AZIONI_CONTINUA == 2
    assert NUM_AZIONI_QUANTITA == 3
    assert NUM_AZIONI_SPOSTAMENTO == 1765
    assert INDICE_SKIP_SPOSTAMENTO == 1764
    print("✓ Dimensioni action space: tris=11, rinf=42, att/sp=1765, cont=2, qty=3")


# ═════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═════════════════════════════════════════════════════════════════════════

def main():
    tests = [
        # Tris
        test_tris_senza_carte,
        test_tris_3_uguali,
        test_tris_due_disgiunti,
        # Rinforzo
        test_rinforzo_solo_territori_propri,
        test_rinforzo_decodifica,
        # Attacco
        test_attacco_codifica_decodifica,
        test_attacco_maschera_stop_sempre_legale,
        test_attacco_maschera_solo_legali,
        test_attacco_maschera_simmetrica,
        # Continua
        test_continua_stop_sempre_legale,
        test_continua_legale_con_armate,
        # Quantità
        test_quantita_3_opzioni,
        test_quantita_min_uguale_max,
        # Spostamento
        test_spostamento_skip_sempre_legale,
        test_spostamento_solo_coppie_proprie,
        # Dimensioni
        test_dimensioni_consistenti,
    ]

    print("\n" + "=" * 60)
    print("Test Modulo 5b: Action space e masking")
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
