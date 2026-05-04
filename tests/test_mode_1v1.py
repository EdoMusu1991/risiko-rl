"""
test_mode_1v1.py — Test della modalità 1v1 (per AlphaZero MCTS prototipo).

Verifica:
- Setup corretto: solo BLU e ROSSO vivi, ~21 territori ognuno
- Snapshot/restore funziona in 1v1
- Reward margin coerente
- Partita 1v1 completa va a fine senza errori
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env import encoding as _encoding
_encoding.STAGE_A_ATTIVO = False
from risiko_env import RisikoEnv


def test_setup_1v1():
    """Solo BLU e ROSSO vivi, VERDE e GIALLO morti."""
    env = RisikoEnv(seed=42, mode_1v1=True)
    obs, info = env.reset()

    vivi = [c for c, g in env.stato.giocatori.items() if g.vivo]
    assert vivi == ["BLU", "ROSSO"] or sorted(vivi) == ["BLU", "ROSSO"], (
        f"Vivi: {vivi}, atteso solo BLU e ROSSO"
    )

    morti = [c for c, g in env.stato.giocatori.items() if not g.vivo]
    assert sorted(morti) == ["GIALLO", "VERDE"], f"Morti: {morti}"


def test_distribuzione_territori_1v1():
    """Territori ridistribuiti fra BLU e ROSSO (~21 ciascuno)."""
    for seed in range(10):
        env = RisikoEnv(seed=seed, mode_1v1=True)
        env.reset()
        n_blu = sum(1 for t in env.stato.mappa.values() if t.proprietario == "BLU")
        n_rosso = sum(1 for t in env.stato.mappa.values() if t.proprietario == "ROSSO")
        n_morti = sum(1 for t in env.stato.mappa.values()
                      if t.proprietario in ("VERDE", "GIALLO"))

        assert n_blu + n_rosso == 42, f"seed={seed}: BLU+ROSSO = {n_blu+n_rosso}, atteso 42"
        assert n_morti == 0, f"seed={seed}: territori morti = {n_morti}"
        # Bilanciato: max 1 di differenza
        assert abs(n_blu - n_rosso) <= 1, f"seed={seed}: BLU={n_blu}, ROSSO={n_rosso}"


def test_morti_no_carte():
    """VERDE e GIALLO non devono avere carte (rimosse al setup)."""
    env = RisikoEnv(seed=42, mode_1v1=True)
    env.reset()
    for c in ("VERDE", "GIALLO"):
        carte = env.stato.giocatori[c].carte
        assert len(carte) == 0, f"{c} ha {len(carte)} carte, atteso 0"


def test_partita_1v1_termina():
    """Una partita 1v1 con bot random termina in <60 round."""
    import random as _random
    n_terminate = 0
    n_max_round = 0
    for seed in range(10):
        env = RisikoEnv(seed=seed, mode_1v1=True)
        obs, info = env.reset()
        rng = _random.Random(seed)
        while True:
            if info is None:
                break
            mask = info["action_mask"]
            legali = np.where(mask)[0]
            if len(legali) == 0:
                break
            obs, r, t, tr, info = env.step(int(rng.choice(legali)))
            if t or tr:
                break
        n_terminate += 1
        n_max_round = max(n_max_round, env.stato.round_corrente)

    assert n_terminate == 10, f"Solo {n_terminate}/10 partite terminate"
    assert n_max_round <= 60, f"Max round = {n_max_round}, cap atteso 60"


def test_reward_margin_calcolato():
    """Reward margin restituisce un float in [-1, +1]."""
    import random as _random
    for seed in range(5):
        env = RisikoEnv(seed=seed, mode_1v1=True, reward_mode="margin")
        obs, info = env.reset()
        rng = _random.Random(seed)
        ultimo_reward = 0
        while True:
            if info is None:
                break
            mask = info["action_mask"]
            legali = np.where(mask)[0]
            if len(legali) == 0:
                break
            obs, r, t, tr, info = env.step(int(rng.choice(legali)))
            ultimo_reward = r
            if t or tr:
                break
        assert -1.0 <= ultimo_reward <= 1.0, (
            f"seed={seed}: reward = {ultimo_reward}, atteso in [-1, +1]"
        )


def test_reward_binary_invariato():
    """In modalita binary, reward in {-1, -0.3, 0.3, +1}."""
    import random as _random
    valori_visti = set()
    for seed in range(20):
        env = RisikoEnv(seed=seed, mode_1v1=True, reward_mode="binary")
        obs, info = env.reset()
        rng = _random.Random(seed)
        while True:
            if info is None:
                break
            mask = info["action_mask"]
            legali = np.where(mask)[0]
            if len(legali) == 0:
                break
            obs, r, t, tr, info = env.step(int(rng.choice(legali)))
            if t or tr:
                break
        valori_visti.add(round(r, 2))
    # Almeno alcuni dei 4 valori standard
    assert valori_visti.issubset({-1.0, -0.3, 0.3, 1.0}), (
        f"Valori reward inattesi: {valori_visti}"
    )


def test_snapshot_restore_in_1v1():
    """Snapshot/restore funziona anche in modalita 1v1."""
    env = RisikoEnv(seed=42, mode_1v1=True)
    obs, info = env.reset()

    # Avanza alcuni step
    rng = np.random.RandomState(0)
    for _ in range(15):
        if info is None:
            break
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        if len(legali) == 0:
            break
        obs, r, t, tr, info = env.step(int(rng.choice(legali)))
        if t or tr:
            break

    snap = env.snapshot()
    s_pre = (env.stato.round_corrente, env.stato.giocatore_corrente)

    # Simula
    for _ in range(50):
        if info is None:
            break
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        if len(legali) == 0:
            break
        obs, r, t, tr, info = env.step(int(rng.choice(legali)))
        if t or tr:
            break

    env.restore(snap)
    s_post = (env.stato.round_corrente, env.stato.giocatore_corrente)
    assert s_pre == s_post, "Snapshot/restore in 1v1 FALLITO"


def test_4player_invariato():
    """Modalita default (4 player) deve funzionare come prima."""
    env = RisikoEnv(seed=42)  # mode_1v1=False di default
    env.reset()
    vivi = [c for c, g in env.stato.giocatori.items() if g.vivo]
    assert sorted(vivi) == ["BLU", "GIALLO", "ROSSO", "VERDE"], (
        f"Mode default rotta: vivi = {vivi}"
    )


# ────────────────────────────────────────────────────────────────────────
#  RUNNER
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Setup 1v1: solo BLU+ROSSO vivi", test_setup_1v1),
        ("Distribuzione territori 1v1 (~21 ciascuno)", test_distribuzione_territori_1v1),
        ("VERDE+GIALLO senza carte", test_morti_no_carte),
        ("Partita 1v1 termina in <=60 round", test_partita_1v1_termina),
        ("Reward margin in [-1, +1]", test_reward_margin_calcolato),
        ("Reward binary invariato", test_reward_binary_invariato),
        ("Snapshot/restore in 1v1", test_snapshot_restore_in_1v1),
        ("4-player default invariato (no regressione)", test_4player_invariato),
    ]

    passati = 0
    falliti = []
    for nome, fn in tests:
        try:
            fn()
            print(f"  [OK] {nome}")
            passati += 1
        except AssertionError as e:
            print(f"  [FAIL] {nome}: {e}")
            falliti.append((nome, str(e)))
        except Exception as e:
            print(f"  [ERROR] {nome}: {type(e).__name__}: {e}")
            falliti.append((nome, f"{type(e).__name__}: {e}"))

    print()
    if falliti:
        print(f"FALLITI {len(falliti)}/{len(tests)}:")
        for nome, msg in falliti:
            print(f"  - {nome}")
        sys.exit(1)
    else:
        print(f"TUTTI I {len(tests)} TEST PASSATI ✓")
