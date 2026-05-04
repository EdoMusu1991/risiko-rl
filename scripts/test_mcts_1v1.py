"""
test_mcts_1v1.py — Test MCTS bot vs random bot in modalita' 1v1.

Specifica ChatGPT (Settimana 2):
> "Misura: tempo per mossa, winrate vs random"
> Se gia' a 150 simulazioni: batti random 70% -> sei sulla strada giusta

NB: NON e' un test della suite (richiede ~5 minuti).
Lancialo manualmente: python scripts/test_mcts_1v1.py [--n_partite N] [--n_sim N]
"""

import argparse
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: F401

from risiko_env import encoding as _encoding
_encoding.STAGE_A_ATTIVO = False

from risiko_env import RisikoEnv
from mcts import MCTSPlanner


def gioca_partita_mcts_vs_random(seed: int, n_simulazioni: int = 100) -> tuple:
    """
    Gioca una partita 1v1 dove BLU = MCTS bot, ROSSO = random.

    Returns:
        (vincitore, reward_blu, n_round, tempo_totale)
    """
    env = RisikoEnv(seed=seed, mode_1v1=True, reward_mode="margin")
    obs, info = env.reset()
    planner = MCTSPlanner(env, c_puct=1.4, rng_seed=seed)

    n_step_bot = 0
    t0 = time.perf_counter()
    while True:
        if info is None:
            break
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        if len(legali) == 0:
            break

        # Solo 1 azione legale: prendila senza MCTS (risparmio tempo)
        if len(legali) == 1:
            azione = int(legali[0])
        else:
            azione = planner.scegli_azione(n_simulazioni=n_simulazioni)

        obs, reward, term, trunc, info = env.step(int(azione))
        n_step_bot += 1
        if term or trunc:
            break

    elapsed = time.perf_counter() - t0
    vinc = info.get("vincitore") if info else None
    return (vinc, float(reward), env.stato.round_corrente, elapsed, n_step_bot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_partite", type=int, default=20)
    parser.add_argument("--n_sim", type=int, default=100,
                        help="Simulazioni MCTS per mossa")
    args = parser.parse_args()

    print(f"Test MCTS vs random bot in 1v1")
    print(f"Partite: {args.n_partite}, simulazioni MCTS: {args.n_sim}")
    print()

    n_blu_vince = 0
    rewards = []
    tempi = []
    rounds = []
    n_step_medi = []

    for seed in range(args.n_partite):
        vinc, reward, n_round, elapsed, n_step = gioca_partita_mcts_vs_random(
            seed=seed, n_simulazioni=args.n_sim
        )
        rewards.append(reward)
        tempi.append(elapsed)
        rounds.append(n_round)
        n_step_medi.append(n_step)
        if vinc == "BLU":
            n_blu_vince += 1

        if (seed + 1) % 5 == 0:
            wr_so_far = n_blu_vince / (seed + 1) * 100
            print(f"  Partita {seed+1}/{args.n_partite}: vinc={vinc} reward={reward:+.2f} "
                  f"({elapsed:.1f}s, {n_step} step bot) | WR finora: {wr_so_far:.0f}%")

    wr = n_blu_vince / args.n_partite * 100

    print()
    print("=" * 60)
    print(f"RISULTATO: MCTS vs random bot in 1v1")
    print("=" * 60)
    print(f"  Win rate BLU (MCTS):  {wr:.1f}% ({n_blu_vince}/{args.n_partite})")
    print(f"  Reward medio:         {np.mean(rewards):+.3f}")
    print(f"  Tempo medio/partita:  {np.mean(tempi):.1f}s")
    print(f"  Step bot medi:        {np.mean(n_step_medi):.0f}")
    print(f"  Round medi:           {np.mean(rounds):.1f}")
    print()
    print(f"Verdetto (criterio ChatGPT):")
    if wr >= 70:
        print(f"  WR >= 70%: ✓ MCTS sulla strada giusta")
    elif wr >= 50:
        print(f"  50% <= WR < 70%: MCTS funziona ma puo' migliorare")
    else:
        print(f"  WR < 50%: ✗ Problema strutturale, MCTS non funziona bene")


if __name__ == "__main__":
    main()
