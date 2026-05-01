"""
batch_eval.py — Valuta in batch N checkpoint e produce tabella ordinata.

Utile dopo training lunghi: invece di valutare manualmente checkpoint per
checkpoint, passi un pattern e ti dice qual è il migliore.

Uso:
    python scripts/batch_eval.py "C:\\path\\to\\folder\\*.zip"
    python scripts/batch_eval.py "C:\\Users\\me\\Downloads\\risiko_bot_*_steps.zip" --n_partite 100
    python scripts/batch_eval.py "C:\\path\\models\\" --n_partite 200

Output: tabella ordinata per win rate (stesso seed_base per confronti puliti).
"""

import argparse
import sys
import os
import glob
import re
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: F401 (auto-applica fix UTF-8)

from risiko_env import RisikoEnv


def estrai_step(filename: str) -> int:
    """Estrae il numero di step dal nome del file. Restituisce -1 se non trovato."""
    match = re.search(r"_(\d+)_steps", filename)
    if match:
        return int(match.group(1))
    if "finale" in filename.lower():
        return 999_999_999  # finale ordinato per ultimo
    return -1


def valuta_modello(path: str, n_partite: int, seed_base: int) -> dict:
    """Valuta un singolo modello, restituisce statistiche."""
    from _helpers import carica_modello_con_autodetect
    model = carica_modello_con_autodetect(path, verbose=False)

    n_vinte = 0
    rewards = []
    posizioni = []
    eliminati = 0
    durate = []

    for seed in range(n_partite):
        env = RisikoEnv(seed=seed_base + seed)
        obs, info = env.reset()
        n_step = 0
        while True:
            mask = info["action_mask"]
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            n_step += 1
            if term or trunc:
                break

        if reward == 1.0:
            n_vinte += 1
        rewards.append(reward)
        durate.append(n_step)
        if not env.stato.giocatori["BLU"].vivo:
            eliminati += 1

        # Posizione finale (1=vince, 4=ultimo/eliminato)
        if reward == 1.0:
            posizioni.append(1)
        elif reward == 0.3:
            posizioni.append(2)
        elif reward == -0.3:
            posizioni.append(3)
        else:
            posizioni.append(4)

    return {
        "win_rate": n_vinte / n_partite,
        "reward_medio": float(np.mean(rewards)),
        "step_medio": float(np.mean(durate)),
        "elim_pct": eliminati / n_partite,
        "n_4": sum(1 for p in posizioni if p == 4) / n_partite,
        "n_partite": n_partite,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch eval di checkpoint.")
    parser.add_argument("pattern",
                        help="Pattern glob (es: 'C:\\path\\*.zip') o cartella")
    parser.add_argument("--n_partite", type=int, default=100,
                        help="Partite per checkpoint nello scan iniziale (default 100)")
    parser.add_argument("--seed_base", type=int, default=0,
                        help="Seed base (uguale per tutti = confronto pulito)")
    parser.add_argument("--top", type=int, default=10,
                        help="Mostra solo i top N (default 10)")
    parser.add_argument("--validate_top", type=int, default=0,
                        help="Dopo lo scan, rivaluta i top N con più partite (default 0=skip)")
    parser.add_argument("--n_partite_validate", type=int, default=500,
                        help="Partite per la validazione (default 500)")
    args = parser.parse_args()

    # Risolvi pattern
    if os.path.isdir(args.pattern):
        files = sorted(glob.glob(os.path.join(args.pattern, "*.zip")))
    else:
        files = sorted(glob.glob(args.pattern))

    if not files:
        print(f"ERRORE: nessun file trovato per pattern: {args.pattern}")
        sys.exit(1)

    # Ordina per step
    files.sort(key=lambda f: estrai_step(os.path.basename(f)))

    print(f"\nValutazione di {len(files)} modelli, {args.n_partite} partite ciascuno")
    print(f"(stesso seed_base = {args.seed_base} per tutti, confronto pulito)\n")
    print(f"{'Step':>10} {'Modello':<40} {'WR':>7} {'Reward':>8} {'Elim':>7} {'4°':>7} {'Tempo':>7}")
    print("-" * 100)

    risultati = []
    for f in files:
        nome = os.path.basename(f)
        step = estrai_step(nome)
        step_str = f"{step:,}" if step > 0 and step < 999_000_000 else "finale"

        t_start = time.time()
        try:
            stats = valuta_modello(f, args.n_partite, args.seed_base)
            durata = time.time() - t_start
            risultati.append({"file": f, "nome": nome, "step": step, **stats,
                              "durata_s": durata})
            print(f"{step_str:>10} {nome:<40} "
                  f"{stats['win_rate']*100:>6.1f}% "
                  f"{stats['reward_medio']:>+8.3f} "
                  f"{stats['elim_pct']*100:>6.1f}% "
                  f"{stats['n_4']*100:>6.1f}% "
                  f"{durata:>6.1f}s")
        except Exception as e:
            print(f"{step_str:>10} {nome:<40} ERRORE: {type(e).__name__}: {e}")

    # Ordina per win rate
    risultati.sort(key=lambda x: -x["win_rate"])

    print()
    print("=" * 100)
    print(f"  TOP {min(args.top, len(risultati))} per win rate")
    print("=" * 100)
    print()
    print(f"{'Rank':>5} {'Step':>10} {'Modello':<40} {'WR':>7} {'Reward':>8} {'Elim':>7}")
    print("-" * 90)
    for i, r in enumerate(risultati[:args.top], 1):
        step_str = f"{r['step']:,}" if 0 < r['step'] < 999_000_000 else "finale"
        print(f"{i:>5} {step_str:>10} {r['nome']:<40} "
              f"{r['win_rate']*100:>6.1f}% "
              f"{r['reward_medio']:>+8.3f} "
              f"{r['elim_pct']*100:>6.1f}%")

    if risultati:
        miglior = risultati[0]
        print(f"\n*** MIGLIORE NELLO SCAN: {miglior['nome']} ***")
        print(f"    Win rate: {miglior['win_rate']*100:.1f}% (su {args.n_partite} partite)")
        print(f"    Path: {miglior['file']}")

    # === VALIDATION DEI TOP N ===
    if args.validate_top > 0 and len(risultati) >= args.validate_top:
        print()
        print("=" * 100)
        print(f"  VALIDAZIONE TOP {args.validate_top} su {args.n_partite_validate} partite")
        print("=" * 100)
        print(f"\nUno scan con N={args.n_partite} ha CI Wilson larghi (~±10%).")
        print(f"Rivaluto i top {args.validate_top} su {args.n_partite_validate} partite per ridurre la varianza.\n")

        validati = []
        for i, r in enumerate(risultati[:args.validate_top], 1):
            step_str = f"{r['step']:,}" if 0 < r['step'] < 999_000_000 else "finale"
            print(f"[{i}/{args.validate_top}] Validazione di {r['nome']} ({step_str} step)...")
            t_start = time.time()
            try:
                stats_v = valuta_modello(r["file"], args.n_partite_validate, args.seed_base)
                durata = time.time() - t_start
                validati.append({**r, **{f"v_{k}": v for k, v in stats_v.items()}, "v_durata_s": durata})

                # Calcola CI Wilson 95%
                from scripts.valuta_completo import wilson_ci
                n_v = stats_v["n_partite"]
                wr_v = stats_v["win_rate"]
                n_vinte_v = int(wr_v * n_v)
                ci_low, ci_high = wilson_ci(n_vinte_v, n_v)

                print(f"    Win rate (scan {args.n_partite}): {r['win_rate']*100:.1f}%")
                print(f"    Win rate (validate {n_v}): {wr_v*100:.1f}%  CI 95%: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
                print(f"    Reward medio: {stats_v['reward_medio']:+.3f}")
                print(f"    Eliminazioni: {stats_v['elim_pct']*100:.1f}%")
                print(f"    4° posto: {stats_v['n_4']*100:.1f}%")
                print(f"    Tempo: {durata:.0f}s")
                print()
            except Exception as e:
                print(f"    ERRORE: {type(e).__name__}: {e}\n")

        # Re-ranking dei validati
        if validati:
            validati.sort(key=lambda x: -x["v_win_rate"])
            print()
            print("=" * 100)
            print(f"  RANKING FINALE (basato su validazione {args.n_partite_validate} partite)")
            print("=" * 100)
            print()
            print(f"{'Rank':>5} {'Step':>10} {'Modello':<40} {'WR scan':>9} {'WR vali':>9} {'CI 95%':>20} {'Reward':>8}")
            print("-" * 110)
            for i, r in enumerate(validati, 1):
                step_str = f"{r['step']:,}" if 0 < r['step'] < 999_000_000 else "finale"
                from scripts.valuta_completo import wilson_ci
                n_v = r["v_n_partite"]
                wr_v = r["v_win_rate"]
                n_vinte_v = int(wr_v * n_v)
                ci_low, ci_high = wilson_ci(n_vinte_v, n_v)
                ci_str = f"[{ci_low*100:.1f}%-{ci_high*100:.1f}%]"
                print(f"{i:>5} {step_str:>10} {r['nome']:<40} "
                      f"{r['win_rate']*100:>8.1f}% "
                      f"{wr_v*100:>8.1f}% "
                      f"{ci_str:>20} "
                      f"{r['v_reward_medio']:>+8.3f}")

            best = validati[0]
            print(f"\n*** MIGLIORE VALIDATO: {best['nome']} ***")
            print(f"    Win rate: {best['v_win_rate']*100:.1f}% (su {best['v_n_partite']} partite)")
            print(f"    Reward medio: {best['v_reward_medio']:+.3f}")
            print(f"    Path: {best['file']}")


if __name__ == "__main__":
    main()
