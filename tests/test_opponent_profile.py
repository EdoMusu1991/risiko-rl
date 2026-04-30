"""
test_opponent_profile.py — Test del Modulo Stage A (opponent embedding).

Verifica:
- DIM_OBSERVATION aumentato a 330
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
    """DIM_OBSERVATION deve essere 330 (era 318)."""
    assert DIM_OBSERVATION == 330, f"DIM_OBSERVATION sbagliata: {DIM_OBSERVATION}"
    assert DIM_OPPONENT_PROFILE == 12
    print(f"✓ DIM_OBSERVATION = {DIM_OBSERVATION}, opponent profile = {DIM_OPPONENT_PROFILE}")


def test_codifica_senza_storia_da_zeri():
    """Senza storia, le 12 feature opponent profile devono essere zeri."""
    stato = crea_partita_iniziale(seed=42)
    obs = codifica_osservazione(stato, "BLU", storia_mosse=None)
    assert obs.shape == (DIM_OBSERVATION,)
    profilo = obs[-DIM_OPPONENT_PROFILE:]
    assert np.allclose(profilo, 0.0), f"Profilo dovrebbe essere zeri: {profilo}"
    print("✓ Senza storia: 12 feature opponent profile = 0")


def test_codifica_con_storia_popolata():
    """Con storia, le feature riflettono il comportamento dell'avversario."""
    stato = crea_partita_iniziale(seed=42)
    storia = {
        "ROSSO": [
            {"attaccato": True, "num_attacchi": 2, "attacchi_contro_pov": 2,
             "ratio_medio": 1.5, "territori_conquistati": 1},
            {"attaccato": True, "num_attacchi": 1, "attacchi_contro_pov": 1,
             "ratio_medio": 1.5, "territori_conquistati": 1},
        ],
        "VERDE": [],
        "GIALLO": [],
    }
    obs = codifica_osservazione(stato, "BLU", storia_mosse=storia)
    profilo = obs[-DIM_OPPONENT_PROFILE:]

    # ROSSO: aggressivita=1.0, focus=1.0 (3/3 attacchi contro pov), risk=0.4 ((1.5-0.5)/2.5)
    # ROSSO è il primo avversario di BLU (indici 0-3)
    assert profilo[0] == 1.0, f"Aggressività ROSSO sbagliata: {profilo[0]}"
    assert profilo[1] == 1.0, f"Focus ROSSO sbagliato: {profilo[1]}"
    assert abs(profilo[2] - 0.4) < 0.01, f"Risk tolerance ROSSO sbagliato: {profilo[2]}"
    # VERDE e GIALLO devono essere zero
    assert np.allclose(profilo[4:8], 0.0), "VERDE dovrebbe essere zero"
    assert np.allclose(profilo[8:12], 0.0), "GIALLO dovrebbe essere zero"
    print(f"✓ Storia popolata: ROSSO ha aggressività=1.0, focus=1.0, risk=0.4")


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
    """L'observation prodotta da env contiene le 12 feature finali correttamente."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()
    assert obs.shape == (DIM_OBSERVATION,)
    profilo = obs[-DIM_OPPONENT_PROFILE:]
    # All'inizio storia vuota → profilo zero
    assert np.allclose(profilo, 0.0), f"Profilo iniziale non zero: {profilo}"

    # Esegui un po'
    for _ in range(80):
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        action = np.random.choice(legali)
        obs, reward, term, trunc, info = env.step(int(action))
        if term or trunc:
            break

    profilo = obs[-DIM_OPPONENT_PROFILE:]
    # Ora il profilo dovrebbe essere non-zero (almeno qualche feature)
    assert np.any(profilo > 0), "Profilo ancora zero dopo step"
    # Tutte le feature in [0, 1]
    assert np.all(profilo >= 0) and np.all(profilo <= 1), \
        f"Feature fuori range: {profilo}"
    print(f"✓ Observation contiene profile valido")


def main():
    tests = [
        test_dim_observation_aggiornata,
        test_codifica_senza_storia_da_zeri,
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
