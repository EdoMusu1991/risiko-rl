"""
test_opponent_profile.py — Test del Modulo Stage A (opponent embedding).

Verifica:
- DIM_OBSERVATION aumentato a 342 (Stage A2: 8 feature × 3 avversari = 24)
- Storia mosse popolata correttamente
- Feature opponent profile coerenti
- Storia bounded (max 50 mosse)
- Reset pulisce la storia

Esegui: python tests\test_opponent_profile.py
"""

import sys
import os
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env.encoding import (
    DIM_OBSERVATION,
    DIM_OPPONENT_PROFILE,
    FINESTRA_OPPONENT_PROFILE,
    codifica_osservazione,
    _codifica_opponent_profile,
)
from risiko_env.env import RisikoEnv
from risiko_env.setup import crea_partita_iniziale


def test_dim_observation_aggiornata():
    """DIM_OBSERVATION deve essere 342 (Stage A2: 8 feature × 3 avversari = 24)."""
    assert DIM_OBSERVATION == 342, f"DIM_OBSERVATION sbagliata: {DIM_OBSERVATION}"
    assert DIM_OPPONENT_PROFILE == 24
    print(f"✓ DIM_OBSERVATION = {DIM_OBSERVATION}, opponent profile = {DIM_OPPONENT_PROFILE}")


def test_codifica_senza_storia_riflette_stato():
    """
    Senza storia, le 24 feature riflettono lo stato corrente:
    - feature 1-6 (di stato) hanno valori sensati (territori, armate, ecc.)
    - feature 7-8 (storia) sono zero senza storia.

    Stage A2: anche senza storia, il profilo NON e' tutto zeri perche' include
    feature di stato calcolate dallo stato corrente.
    """
    stato = crea_partita_iniziale(seed=42)
    obs = codifica_osservazione(stato, "BLU", storia_mosse=None)
    assert obs.shape == (DIM_OBSERVATION,)
    profilo = obs[-DIM_OPPONENT_PROFILE:]

    # Stato iniziale: ogni avversario ha 9-11 territori e 30 armate
    # Quindi territori_norm > 0 e armate_norm > 0 per tutti
    for i in range(3):
        feat_avv = profilo[i*8:(i+1)*8]
        terr_norm = feat_avv[0]
        arm_norm = feat_avv[1]
        assert 0.15 < terr_norm < 0.30, f"Avv {i}: territori_norm fuori range: {terr_norm}"
        assert 0.20 < arm_norm < 0.30, f"Avv {i}: armate_norm fuori range: {arm_norm}"

        # Feature 7-8 (storia) devono essere zero senza storia
        cnq_recenti = feat_avv[6]
        per_recenti = feat_avv[7]
        assert cnq_recenti == 0.0, f"Avv {i}: conquiste_recenti dovrebbe essere 0 senza storia"
        assert per_recenti == 0.0, f"Avv {i}: perdite_recenti dovrebbe essere 0 senza storia"

    print("✓ Senza storia: feature di stato OK, feature di storia = 0")


def test_codifica_con_storia_popolata():
    """
    Con storia di conquiste, le feature 7-8 (recency) sono > 0.
    """
    stato = crea_partita_iniziale(seed=42)
    storia = {
        "ROSSO": [
            {"attaccato": True, "num_attacchi": 2, "attacchi_contro_pov": 2,
             "ratio_medio": 1.5, "territori_conquistati": 2, "territori_persi": 0},
            {"attaccato": True, "num_attacchi": 1, "attacchi_contro_pov": 1,
             "ratio_medio": 1.5, "territori_conquistati": 1, "territori_persi": 0},
        ],
        "VERDE": [],
        "GIALLO": [],
    }
    obs = codifica_osservazione(stato, "BLU", storia_mosse=storia)
    profilo = obs[-DIM_OPPONENT_PROFILE:]

    # ROSSO ha conquistato 3 territori in totale negli ultimi 2 turni
    # conquiste_recenti = min(1.0, 3/5) = 0.6
    feat_rosso = profilo[0:8]
    cnq_rosso = feat_rosso[6]
    assert abs(cnq_rosso - 0.6) < 0.01, f"Conquiste recenti ROSSO sbagliate: {cnq_rosso}"

    # VERDE e GIALLO: feature di storia (7,8) devono essere zero
    feat_verde = profilo[8:16]
    feat_giallo = profilo[16:24]
    assert feat_verde[6] == 0.0 and feat_verde[7] == 0.0, "VERDE storia dovrebbe essere zero"
    assert feat_giallo[6] == 0.0 and feat_giallo[7] == 0.0, "GIALLO storia dovrebbe essere zero"

    print(f"✓ Storia popolata: ROSSO conquiste_recenti={cnq_rosso:.2f}")


def test_storia_iniziale_vuota_in_env():
    """Dopo reset(), _storia_mosse deve essere {ROSSO:[], VERDE:[], GIALLO:[]}."""
    env = RisikoEnv(seed=42)
    env.reset()
    assert hasattr(env, "_storia_mosse")
    assert env._storia_mosse == {"ROSSO": [], "VERDE": [], "GIALLO": []}
    print("✓ Storia iniziale vuota dopo reset")


def test_storia_popolata_dopo_step():
    """Dopo alcuni step, la storia deve contenere mosse degli avversari."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()

    n_step = 0
    while n_step < 100:
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        action = np.random.choice(legali)
        obs, reward, term, trunc, info = env.step(int(action))
        n_step += 1
        if term or trunc:
            break

    # Almeno un avversario deve avere mosse registrate
    n_mosse_totali = sum(len(m) for m in env._storia_mosse.values())
    assert n_mosse_totali > 0, "Storia non popolata dopo step"
    # Tutti i campi della prima mossa devono esistere
    for c, mosse in env._storia_mosse.items():
        if mosse:
            m = mosse[0]
            for k in ["turno", "attaccato", "num_attacchi", "attacchi_contro_pov",
                      "ratio_medio", "territori_conquistati"]:
                assert k in m, f"Campo {k} mancante in mossa {m}"
    print(f"✓ Storia popolata dopo {n_step} step: {n_mosse_totali} mosse totali")


def test_storia_bounded():
    """La storia non deve crescere oltre 50 mosse per giocatore."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()

    while True:
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        action = np.random.choice(legali)
        obs, reward, term, trunc, info = env.step(int(action))
        if term or trunc:
            break

    for c, mosse in env._storia_mosse.items():
        assert len(mosse) <= 50, f"Storia {c} troppo lunga: {len(mosse)}"
    print(f"✓ Storia bounded a 50 mosse per giocatore")


def test_reset_pulisce_storia():
    """reset() deve azzerare la storia delle mosse."""
    env = RisikoEnv(seed=42)
    env.reset()
    obs, info = env.reset()
    # Esegui qualche step
    for _ in range(50):
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        action = np.random.choice(legali)
        obs, reward, term, trunc, info = env.step(int(action))
        if term or trunc:
            break

    # Storia dovrebbe essere popolata
    assert any(len(m) > 0 for m in env._storia_mosse.values())

    # Re-reset
    env.reset()
    assert all(len(m) == 0 for m in env._storia_mosse.values()), \
        f"Storia non pulita: {env._storia_mosse}"
    print("✓ reset() azzera storia")


def test_observation_inserisce_profile_correttamente():
    """L'observation prodotta da env contiene le 24 feature Stage A2."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()
    assert obs.shape == (DIM_OBSERVATION,)
    profilo = obs[-DIM_OPPONENT_PROFILE:]
    # Stage A2: anche all'inizio il profilo non e' zero (feature di stato attive)
    # Le feature 1-6 (stato) hanno gia' valori. Le feature 7-8 (storia) sono zero.
    # Verifica: per ogni avversario, almeno una feature di stato e' non-zero.
    for i in range(3):
        feat = profilo[i*8:(i+1)*8]
        feat_stato = feat[:6]  # solo le 6 feature di stato
        assert np.any(feat_stato > 0), f"Avv {i}: feature di stato tutte zero all'inizio"
        # Feature 7-8 (storia) sono zero senza storia
        assert feat[6] == 0.0 and feat[7] == 0.0, f"Avv {i}: feature di storia non zero"

    # Esegui un po'
    for _ in range(80):
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        action = np.random.choice(legali)
        obs, reward, term, trunc, info = env.step(int(action))
        if term or trunc:
            break

    profilo = obs[-DIM_OPPONENT_PROFILE:]
    # Ora il profilo dovrebbe essere non-zero
    assert np.any(profilo > 0), "Profilo ancora zero dopo step"
    # Tutte le feature in [0, 1]
    assert np.all(profilo >= 0) and np.all(profilo <= 1), \
        f"Feature fuori range: {profilo}"
    print(f"✓ Observation contiene profile valido")


def main():
    tests = [
        test_dim_observation_aggiornata,
        test_codifica_senza_storia_riflette_stato,
        test_codifica_con_storia_popolata,
        test_storia_iniziale_vuota_in_env,
        test_storia_popolata_dopo_step,
        test_storia_bounded,
        test_reset_pulisce_storia,
        test_observation_inserisce_profile_correttamente,
    ]

    print("\n" + "=" * 60)
    print("Test Stage A: Opponent Profile")
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
