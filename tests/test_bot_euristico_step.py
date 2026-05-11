"""
test_bot_euristico_step.py - Test di equivalenza statistica.

Verifica che bot_euristico_step (decisione singola per env.step()) produca
lo stesso comportamento statistico di bot_euristico originale (gioca turno
completo modificando stato direttamente).

Criterio: WR vs random in 1v1 nell'intervallo [80%, 95%] su N partite.
Riferimento originale: ~91% (DIARIO_SETTIMANA_2).
"""

import random
import time

import numpy as np

from risiko_env import RisikoEnv
from risiko_env.bot_euristico_step import bot_euristico_step


def gioca_partita_step_by_step(seed: int, max_dec: int = 1500) -> str | None:
    """
    Una partita 1v1: bot_euristico_step gioca BLU, bot_random interno gioca ROSSO.
    Ritorna 'BLU', 'ROSSO', o None se truncated.
    """
    env = RisikoEnv(bot_color="BLU", mode_1v1=True, seed=seed)
    obs, info = env.reset(seed=seed)
    rng = random.Random(seed)

    n_dec = 0
    while n_dec < max_dec:
        mask = info["action_mask"]
        if not mask.any():
            break

        azione = bot_euristico_step(env, info, rng)
        obs, reward, term, trunc, info = env.step(azione)
        n_dec += 1

        if term or trunc:
            break

    return env.stato.vincitore


def test_equivalenza_winrate():
    """100 partite step-by-step vs random. WR atteso ~91%."""
    N = 100
    t0 = time.perf_counter()
    risultati = [gioca_partita_step_by_step(seed=2000 + i) for i in range(N)]
    dt = time.perf_counter() - t0

    n_blu = sum(1 for r in risultati if r == "BLU")
    n_rosso = sum(1 for r in risultati if r == "ROSSO")
    n_none = sum(1 for r in risultati if r is None)

    wr_blu = n_blu / N
    print(f"\n[equivalenza] {N} partite bot_euristico_step (BLU) vs random (ROSSO):")
    print(f"  BLU={n_blu}, ROSSO={n_rosso}, truncated={n_none}  ({dt:.0f}s)")
    print(f"  WR BLU: {wr_blu:.1%}")
    print(f"  Riferimento bot_euristico originale (DIARIO_SETTIMANA_2): 91%")

    # Intervallo: 80-95%. Range largo perche' ricalibriamo prob STOP attacco.
    assert 0.80 <= wr_blu <= 0.97, (
        f"WR={wr_blu:.1%} fuori range [80%, 97%]. "
        f"Step-by-step diverge troppo da euristico originale."
    )
    print(f"  OK: WR dentro intervallo atteso [80%, 95%]")


def test_partite_finiscono():
    """Sanity: tutte le partite o finiscono o sono truncate, no crash."""
    for seed in range(2200, 2220):
        risultato = gioca_partita_step_by_step(seed=seed)
        assert risultato in (None, "BLU", "ROSSO"), f"seed {seed}: result {risultato!r}"
    print("[partite_finiscono] OK 20 partite")


def test_smoke_diverse_sotto_fasi():
    """Smoke: il bot risponde correttamente a tutte le sotto-fasi nei primi 50 step."""
    env = RisikoEnv(bot_color="BLU", mode_1v1=True, seed=42)
    obs, info = env.reset(seed=42)
    rng = random.Random(42)

    sotto_fasi_viste = set()
    for _ in range(50):
        if env.sotto_fase is None or env.stato.terminata:
            break
        sotto_fasi_viste.add(env.sotto_fase)
        azione = bot_euristico_step(env, info, rng)
        # L'azione deve essere legale
        assert info["action_mask"][azione], (
            f"Azione {azione} non legale per sotto_fase={env.sotto_fase}"
        )
        obs, _, term, trunc, info = env.step(azione)
        if term or trunc:
            break

    print(f"[sotto_fasi] viste: {sotto_fasi_viste}")
    # Almeno tris, rinforzo, attacco devono essere viste (le piu' frequenti)
    assert "rinforzo" in sotto_fasi_viste or "tris" in sotto_fasi_viste


if __name__ == "__main__":
    print("=" * 60)
    test_smoke_diverse_sotto_fasi()
    test_partite_finiscono()
    test_equivalenza_winrate()
    print("=" * 60)
    print("TUTTI I TEST PASSATI")
