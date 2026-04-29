"""
test_encoding.py — Test del Modulo 5a (encoding observation).

Verifica:
- Dimensione vettore observation costante e corretta
- Encoding mappa: one-hot proprietario, armate normalizzate, flag obiettivo
- Encoding obiettivo: one-hot corretto
- Encoding carte: conteggio per simbolo
- Encoding avversari: 3 avversari × 4 features
- Encoding controllo continenti: chi possiede ogni continente
- Encoding fase: round, fase, conquiste
- Privacy: il bot non vede gli obiettivi degli avversari né le loro carte specifiche

Esegui: python tests\test_encoding.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env.data import (
    OBIETTIVI,
    COLORI_GIOCATORI,
    CONTINENTI,
    FANTE,
    CANNONE,
    CAVALLO,
    JOLLY,
)
from risiko_env.stato import StatoPartita, Carta
from risiko_env.setup import crea_partita_iniziale
from risiko_env.encoding import (
    codifica_osservazione,
    DIM_OBSERVATION,
    DIM_MAPPA,
    DIM_OBIETTIVO_PROPRIO,
    DIM_CARTE_PROPRIE,
    DIM_AVVERSARI,
    DIM_CONTROLLO_CONTINENTI,
    DIM_FASE_E_TURNO,
    DIM_TRIS_GIOCATI,
    NUM_TERRITORI,
    NUM_OBIETTIVI,
    TERRITORIO_INDEX,
    COLORE_INDEX,
    FASI_ORDINE,
)


# ─────────────────────────────────────────────────────────────────────────
#  TEST DIMENSIONI
# ─────────────────────────────────────────────────────────────────────────

def test_dimensione_observation():
    """L'observation deve avere dimensione costante."""
    stato = crea_partita_iniziale(seed=42)
    obs = codifica_osservazione(stato, "BLU")
    assert obs.shape == (DIM_OBSERVATION,), f"Shape: {obs.shape}"
    assert obs.dtype == np.float32
    print(f"✓ Observation shape ({DIM_OBSERVATION},) dtype float32")


def test_dimensione_componenti():
    """Verifica che la somma delle componenti = DIM_OBSERVATION."""
    somma = (DIM_MAPPA + DIM_OBIETTIVO_PROPRIO + DIM_CARTE_PROPRIE
             + DIM_AVVERSARI + DIM_CONTROLLO_CONTINENTI
             + DIM_FASE_E_TURNO + DIM_TRIS_GIOCATI)
    assert somma == DIM_OBSERVATION, f"Somma {somma} != {DIM_OBSERVATION}"
    print(f"✓ Componenti: mappa={DIM_MAPPA}, obj={DIM_OBIETTIVO_PROPRIO}, "
          f"carte={DIM_CARTE_PROPRIE}, avv={DIM_AVVERSARI}, "
          f"cont={DIM_CONTROLLO_CONTINENTI}, fase={DIM_FASE_E_TURNO}, "
          f"tris={DIM_TRIS_GIOCATI}, totale={DIM_OBSERVATION}")


def test_dimensione_costante_su_seed_diversi():
    """La dimensione deve essere costante indipendentemente dal seed."""
    for seed in range(20):
        stato = crea_partita_iniziale(seed=seed)
        for col in COLORI_GIOCATORI:
            obs = codifica_osservazione(stato, col)
            assert obs.shape == (DIM_OBSERVATION,)
    print("✓ Dimensione costante su 80 (seed × giocatore) combinazioni")


# ─────────────────────────────────────────────────────────────────────────
#  TEST ENCODING MAPPA
# ─────────────────────────────────────────────────────────────────────────

def test_encoding_mappa_proprietario_one_hot():
    """One-hot del proprietario è corretto: esattamente 1 valore a 1.0 per territorio."""
    stato = crea_partita_iniziale(seed=42)
    obs = codifica_osservazione(stato, "BLU")
    # Sezione mappa: i primi DIM_MAPPA elementi
    mappa_section = obs[:DIM_MAPPA].reshape(NUM_TERRITORI, 6)

    for idx in range(NUM_TERRITORI):
        proprietario_oh = mappa_section[idx, :4]
        # Esattamente uno dovrebbe essere 1.0 (visto che ogni territorio ha proprietario)
        assert int(np.sum(proprietario_oh)) == 1, (
            f"Territorio idx={idx}: somma one-hot proprietario = {np.sum(proprietario_oh)}"
        )
    print("✓ Mappa: ogni territorio ha esattamente 1 proprietario one-hot")


def test_encoding_mappa_armate_normalizzate():
    """Armate sono nel range [0, 1]."""
    stato = crea_partita_iniziale(seed=42)
    obs = codifica_osservazione(stato, "BLU")
    mappa_section = obs[:DIM_MAPPA].reshape(NUM_TERRITORI, 6)

    for idx in range(NUM_TERRITORI):
        armate_norm = mappa_section[idx, 4]
        assert 0.0 <= armate_norm <= 1.0, f"Armate norm fuori range: {armate_norm}"
    print("✓ Mappa: armate normalizzate in [0, 1]")


def test_encoding_mappa_flag_in_obiettivo():
    """
    La sesta feature di ogni territorio è 1 se nel proprio obiettivo,
    0 altrimenti. Diverso per ogni POV.
    """
    stato = crea_partita_iniziale(seed=42)

    # POV: Blu vede il proprio obiettivo
    obs_blu = codifica_osservazione(stato, "BLU")
    mappa_blu = obs_blu[:DIM_MAPPA].reshape(NUM_TERRITORI, 6)
    obj_blu = stato.giocatori["BLU"].obiettivo_id
    territori_obj_blu = OBIETTIVI[obj_blu]["territori"]

    for t, idx in TERRITORIO_INDEX.items():
        flag = mappa_blu[idx, 5]
        atteso = 1.0 if t in territori_obj_blu else 0.0
        assert flag == atteso, f"Territorio {t} flag={flag}, atteso {atteso}"

    # POV: Rosso vede il PROPRIO obiettivo, NON quello di Blu
    obs_rosso = codifica_osservazione(stato, "ROSSO")
    mappa_rosso = obs_rosso[:DIM_MAPPA].reshape(NUM_TERRITORI, 6)
    obj_rosso = stato.giocatori["ROSSO"].obiettivo_id

    if obj_rosso != obj_blu:
        # Almeno qualche flag deve essere diverso
        flag_blu = mappa_blu[:, 5]
        flag_rosso = mappa_rosso[:, 5]
        assert not np.array_equal(flag_blu, flag_rosso), (
            "I flag obiettivo dovrebbero differire tra POV diversi"
        )

    print("✓ Mappa: flag in_obiettivo riflette il POV del giocatore")


# ─────────────────────────────────────────────────────────────────────────
#  TEST ENCODING OBIETTIVO PROPRIO
# ─────────────────────────────────────────────────────────────────────────

def test_obiettivo_one_hot():
    """Esattamente 1 valore a 1.0 nella sezione obiettivo proprio."""
    stato = crea_partita_iniziale(seed=42)
    obs = codifica_osservazione(stato, "BLU")
    inizio = DIM_MAPPA
    fine = DIM_MAPPA + DIM_OBIETTIVO_PROPRIO
    obj_section = obs[inizio:fine]

    assert int(np.sum(obj_section)) == 1, f"Somma obiettivo OH: {np.sum(obj_section)}"

    # L'indice dell'1 corrisponde all'obiettivo - 1 (0-indexed)
    obj_id = stato.giocatori["BLU"].obiettivo_id
    expected_idx = obj_id - 1
    actual_idx = int(np.argmax(obj_section))
    assert actual_idx == expected_idx, f"Idx OH: {actual_idx}, atteso {expected_idx}"
    print(f"✓ Obiettivo proprio one-hot: id={obj_id} → idx={actual_idx}")


# ─────────────────────────────────────────────────────────────────────────
#  TEST ENCODING CARTE
# ─────────────────────────────────────────────────────────────────────────

def test_encoding_carte_vuoto():
    """Inizio partita: 0 carte per tutti."""
    stato = crea_partita_iniziale(seed=42)
    obs = codifica_osservazione(stato, "BLU")
    inizio = DIM_MAPPA + DIM_OBIETTIVO_PROPRIO
    fine = inizio + DIM_CARTE_PROPRIE
    carte = obs[inizio:fine]
    # [n_fanti=0, n_cannoni=0, n_cavalli=0, n_jolly=0, totale_norm=0]
    assert np.array_equal(carte, np.zeros(5))
    print("✓ Carte iniziali: tutte a zero")


def test_encoding_carte_con_mano():
    """Mano con vari simboli: encoding corretto."""
    stato = crea_partita_iniziale(seed=42)
    # Aggiungi 3 fanti, 2 cannoni, 1 jolly a Blu
    stato.giocatori["BLU"].carte = [
        Carta("alaska", FANTE),
        Carta("brasile", FANTE),
        Carta("egitto", FANTE),
        Carta("cina", CANNONE),
        Carta("ucraina", CANNONE),
        Carta(None, JOLLY),
    ]
    obs = codifica_osservazione(stato, "BLU")
    inizio = DIM_MAPPA + DIM_OBIETTIVO_PROPRIO
    fine = inizio + DIM_CARTE_PROPRIE
    carte = obs[inizio:fine]
    assert carte[0] == 3.0  # fanti
    assert carte[1] == 2.0  # cannoni
    assert carte[2] == 0.0  # cavalli
    assert carte[3] == 1.0  # jolly
    assert abs(carte[4] - 6/7) < 1e-5  # totale normalizzato (6 carte / 7)
    print("✓ Carte: 3 fanti + 2 cannoni + 1 jolly = encoding corretto")


# ─────────────────────────────────────────────────────────────────────────
#  TEST PRIVACY
# ─────────────────────────────────────────────────────────────────────────

def test_privacy_obiettivi_avversari():
    """
    L'observation di Blu non deve dipendere dagli OBIETTIVI degli avversari.
    Test: cambia gli obiettivi di Rosso/Verde/Giallo, l'observation di Blu
    deve rimanere identica.
    """
    stato1 = crea_partita_iniziale(seed=42)
    obs_blu_1 = codifica_osservazione(stato1, "BLU")

    # Cambia obiettivi degli altri (forziamo)
    stato2 = crea_partita_iniziale(seed=42)
    stato2.giocatori["ROSSO"].obiettivo_id = 1
    stato2.giocatori["VERDE"].obiettivo_id = 2
    stato2.giocatori["GIALLO"].obiettivo_id = 3
    obs_blu_2 = codifica_osservazione(stato2, "BLU")

    assert np.array_equal(obs_blu_1, obs_blu_2), (
        "L'observation di Blu non dovrebbe cambiare se cambiano gli obiettivi avversari"
    )
    print("✓ Privacy: l'observation NON dipende dagli obiettivi avversari")


def test_privacy_carte_specifiche_avversari():
    """
    L'observation di Blu non deve dipendere dalle CARTE SPECIFICHE degli avversari,
    solo dal loro NUMERO di carte.
    """
    stato1 = crea_partita_iniziale(seed=42)
    # Dai a Rosso 3 carte: 3 fanti
    stato1.giocatori["ROSSO"].carte = [
        Carta("alaska", FANTE), Carta("brasile", FANTE), Carta("cina", FANTE),
    ]
    obs_blu_1 = codifica_osservazione(stato1, "BLU")

    # Setup identico ma Rosso ha 3 cavalli (stesso numero, simboli diversi)
    stato2 = crea_partita_iniziale(seed=42)
    stato2.giocatori["ROSSO"].carte = [
        Carta("alaska", CAVALLO), Carta("brasile", CAVALLO), Carta("cina", CAVALLO),
    ]
    obs_blu_2 = codifica_osservazione(stato2, "BLU")

    assert np.array_equal(obs_blu_1, obs_blu_2), (
        "L'observation di Blu non dovrebbe distinguere i simboli specifici delle carte avversarie"
    )
    print("✓ Privacy: l'observation di Blu vede solo il NUMERO carte avversarie, non i simboli")


# ─────────────────────────────────────────────────────────────────────────
#  TEST CAMBI DI POV
# ─────────────────────────────────────────────────────────────────────────

def test_pov_diverse_observations():
    """Diversi POV → diverse observations (almeno in qualche feature)."""
    stato = crea_partita_iniziale(seed=42)
    obs_blu = codifica_osservazione(stato, "BLU")
    obs_rosso = codifica_osservazione(stato, "ROSSO")
    assert not np.array_equal(obs_blu, obs_rosso), (
        "Observations da POV diversi dovrebbero differire"
    )
    print("✓ POV diversi producono observations diverse")


# ─────────────────────────────────────────────────────────────────────────
#  TEST CONTROLLO CONTINENTI
# ─────────────────────────────────────────────────────────────────────────

def test_controllo_continenti_inizio_partita():
    """All'inizio della partita, nessuno controlla un continente intero."""
    stato = crea_partita_iniziale(seed=42)
    obs = codifica_osservazione(stato, "BLU")

    inizio = (DIM_MAPPA + DIM_OBIETTIVO_PROPRIO + DIM_CARTE_PROPRIE
              + DIM_AVVERSARI)
    fine = inizio + DIM_CONTROLLO_CONTINENTI
    controllo = obs[inizio:fine]

    # In molti seed, è possibile (raro) che qualcuno parta con un continente,
    # ma improbabile. Almeno verifichiamo che la sezione sia tutta 0/1
    for v in controllo:
        assert v in (0.0, 1.0), f"Valore controllo continente non binario: {v}"

    # Su 50 seed, conta quante volte qualcuno controlla un continente all'inizio
    contatore = 0
    for seed in range(50):
        s = crea_partita_iniziale(seed=seed)
        o = codifica_osservazione(s, "BLU")
        c = o[inizio:fine]
        if np.sum(c) > 0:
            contatore += 1
    print(f"✓ Controllo continenti binario, all'inizio raro ({contatore}/50 partite)")


def test_controllo_continenti_oceania_tutta():
    """Se Blu controlla tutta l'Oceania, l'encoding lo riflette."""
    stato = StatoPartita()
    # Forza: tutta l'Oceania a Blu
    for t in CONTINENTI["oceania"]:
        stato.mappa[t].proprietario = "BLU"
        stato.mappa[t].armate = 1
    # Resto dei territori a Rosso
    from risiko_env.data import TUTTI_TERRITORI
    for t in TUTTI_TERRITORI:
        if stato.mappa[t].proprietario is None:
            stato.mappa[t].proprietario = "ROSSO"
            stato.mappa[t].armate = 1
    stato.giocatori["BLU"].obiettivo_id = 1
    stato.giocatori["ROSSO"].obiettivo_id = 2
    stato.giocatori["VERDE"].obiettivo_id = 3
    stato.giocatori["GIALLO"].obiettivo_id = 4

    obs = codifica_osservazione(stato, "BLU")
    inizio = (DIM_MAPPA + DIM_OBIETTIVO_PROPRIO + DIM_CARTE_PROPRIE
              + DIM_AVVERSARI)
    fine = inizio + DIM_CONTROLLO_CONTINENTI
    # 6 continenti × 4 colori = 24 features
    controllo = obs[inizio:fine].reshape(6, 4)

    from risiko_env.encoding import CONTINENTE_INDEX
    oceania_idx = CONTINENTE_INDEX["oceania"]
    blu_idx = COLORE_INDEX["BLU"]
    assert controllo[oceania_idx, blu_idx] == 1.0, "Blu non risulta controllare Oceania"
    # Gli altri continenti: Rosso li controlla tutti (perché glielo abbiamo dato)
    for cont, cont_idx in CONTINENTE_INDEX.items():
        if cont == "oceania":
            continue
        rosso_idx = COLORE_INDEX["ROSSO"]
        assert controllo[cont_idx, rosso_idx] == 1.0, f"Rosso non controlla {cont}"
    print("✓ Controllo continenti: Blu controlla Oceania, Rosso controlla gli altri 5")


# ─────────────────────────────────────────────────────────────────────────
#  TEST FASE E TURNO
# ─────────────────────────────────────────────────────────────────────────

def test_encoding_fase():
    """L'encoding della fase corrente è one-hot tra le 4 fasi."""
    stato = crea_partita_iniziale(seed=42)
    inizio = (DIM_MAPPA + DIM_OBIETTIVO_PROPRIO + DIM_CARTE_PROPRIE
              + DIM_AVVERSARI + DIM_CONTROLLO_CONTINENTI)
    fine = inizio + DIM_FASE_E_TURNO

    for fase in FASI_ORDINE:
        obs = codifica_osservazione(stato, "BLU", fase_corrente=fase)
        sezione = obs[inizio:fine]
        # round_norm + 4 one-hot fase + conquiste_norm = 6 valori
        round_norm = sezione[0]
        fase_oh = sezione[1:5]
        conquiste = sezione[5]

        assert round_norm == 1/60.0, f"round_corrente=1, normalizzato {round_norm}"
        assert int(np.sum(fase_oh)) == 1, "Fase deve essere one-hot"
        from risiko_env.encoding import FASE_INDEX
        assert int(np.argmax(fase_oh)) == FASE_INDEX[fase]
    print("✓ Encoding fase: 4 fasi correttamente one-hot")


def test_encoding_round_avanzato():
    """Round=30 → encoding round normalizzato 30/60."""
    stato = crea_partita_iniziale(seed=42)
    stato.round_corrente = 30
    obs = codifica_osservazione(stato, "BLU")
    inizio = (DIM_MAPPA + DIM_OBIETTIVO_PROPRIO + DIM_CARTE_PROPRIE
              + DIM_AVVERSARI + DIM_CONTROLLO_CONTINENTI)
    round_norm = obs[inizio]
    assert abs(round_norm - 0.5) < 1e-5, f"Round 30 → {round_norm}"
    print("✓ Encoding round: 30 → 0.5 (normalizzato)")


# ─────────────────────────────────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_dimensione_observation,
        test_dimensione_componenti,
        test_dimensione_costante_su_seed_diversi,
        test_encoding_mappa_proprietario_one_hot,
        test_encoding_mappa_armate_normalizzate,
        test_encoding_mappa_flag_in_obiettivo,
        test_obiettivo_one_hot,
        test_encoding_carte_vuoto,
        test_encoding_carte_con_mano,
        test_privacy_obiettivi_avversari,
        test_privacy_carte_specifiche_avversari,
        test_pov_diverse_observations,
        test_controllo_continenti_inizio_partita,
        test_controllo_continenti_oceania_tutta,
        test_encoding_fase,
        test_encoding_round_avanzato,
    ]

    print("\n" + "=" * 60)
    print("Test Modulo 5a: Encoding observation")
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
