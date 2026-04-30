"""
auto_eval_durante_training.py — Monitor automatico dei checkpoint durante il training.

Sorveglia una cartella di checkpoint. Ogni volta che vede un nuovo file
.zip, lo valuta con N partite veloci e logga il win rate. Mostra la
progressione del bot in tempo reale.

Particolarmente utile su Colab Pro: lanci il training in background, e in
parallelo lanci questo script per vedere come migliora.

Uso:
    python scripts/auto_eval_durante_training.py /content/drive/MyDrive/risiko-rl-checkpoints
    python scripts/auto_eval_durante_training.py /path/to/checkpoints --n_partite 30
    python scripts/auto_eval_durante_training.py /path/to/checkpoints --polling 60

Output: tabella di progressione (timestep / win rate / reward medio).
Salva anche progressione.csv per grafico futuro.

NB: questo script blocca finché non lo interrompi con Ctrl+C.
"""

import argparse
import sys
import os
import time
import csv
from collections import Counter
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env import RisikoEnv
from risiko_env.data import COLORI_GIOCATORI


def estrai_timestep(filename: str) -> int:
    """Estrae il timestep da un nome file tipo 'risiko_bot_500000_steps.zip'."""
    import re
    match = re.search(r"_(\d+)_steps", filename)
    if match:
        return int(match.group(1))
    return 0


def valuta_veloce(model, n_partite: int = 30, bot_color: str = "BLU") -> dict:
    """Valutazione veloce per progress check."""
    n_vinte = 0
    rewards = []
    durate = []
    for seed in range(n_partite):
        env = RisikoEnv(seed=seed, bot_color=bot_color)
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
    return {
        "win_rate": n_vinte / n_partite,
        "reward_medio": float(np.mean(rewards)),
        "step_medio": float(np.mean(durate)),
        "n_partite": n_partite,
    }


def trova_nuovi_checkpoint(cartella: str, gia_visti: set) -> list:
    """Trova file .zip mai visti prima."""
    if not os.path.exists(cartella):
        return []
    files = os.listdir(cartella)
    nuovi = []
    for f in files:
        if (f.endswith(".zip") and f.startswith("risiko_bot")
                and f not in gia_visti):
            nuovi.append(f)
    return sorted(nuovi, key=estrai_timestep)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor automatico dei checkpoint durante il training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("cartella",
                        help="Cartella dove vengono salvati i checkpoint")
    parser.add_argument("--n_partite", type=int, default=30,
                        help="Partite per valutazione (default 30, veloce)")
    parser.add_argument("--bot_color", default="BLU",
                        choices=COLORI_GIOCATORI)
    parser.add_argument("--polling", type=int, default=60,
                        help="Secondi tra check della cartella (default 60)")
    parser.add_argument("--csv", default="progressione.csv",
                        help="File CSV per progressione (default progressione.csv)")
    args = parser.parse_args()

    if not os.path.exists(args.cartella):
        print(f"ERRORE: cartella non trovata: {args.cartella}")
        print("Suggerimento: crea la cartella o aspetta che il training generi il primo checkpoint")
        sys.exit(1)

    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        raise ImportError("Installa: pip install sb3-contrib")

    print(f"Monitor avviato su: {args.cartella}")
    print(f"Polling ogni {args.polling}s, valutazione su {args.n_partite} partite")
    print(f"Progressione salvata in: {args.csv}")
    print(f"Premi Ctrl+C per fermare\n")

    gia_visti = set()
    progressione = []

    # Header CSV
    csv_esiste = os.path.exists(args.csv)
    if not csv_esiste:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "checkpoint", "timestep", "win_rate",
                       "reward_medio", "step_medio", "n_partite"])

    # Header tabella console
    print(f"{'Time':<10} {'Checkpoint':<35} {'Steps':>10} {'WR':>7} {'Reward':>8} {'Step':>7}")
    print("-" * 80)

    try:
        while True:
            nuovi = trova_nuovi_checkpoint(args.cartella, gia_visti)
            for nome_file in nuovi:
                path = os.path.join(args.cartella, nome_file)
                timestep = estrai_timestep(nome_file)
                try:
                    model = MaskablePPO.load(path)
                    stats = valuta_veloce(model, args.n_partite, args.bot_color)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    wr = stats["win_rate"] * 100
                    r = stats["reward_medio"]
                    s = stats["step_medio"]
                    progressione.append((timestep, wr, r))
                    print(f"{timestamp:<10} {nome_file:<35} {timestep:>10,d} "
                          f"{wr:>6.1f}% {r:>+8.3f} {s:>7.0f}")

                    # Salva CSV
                    with open(args.csv, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([timestamp, nome_file, timestep, f"{stats['win_rate']:.4f}",
                                   f"{r:.4f}", f"{s:.1f}", args.n_partite])

                    gia_visti.add(nome_file)

                    # Mini-grafico testuale ogni 5 valutazioni
                    if len(progressione) >= 2 and len(progressione) % 5 == 0:
                        print(f"\n  Trend ultimi {len(progressione)} checkpoint:")
                        max_wr = max(p[1] for p in progressione)
                        min_wr = min(p[1] for p in progressione)
                        for (ts, wr, _) in progressione[-10:]:
                            range_wr = max(1, max_wr - min_wr)
                            barra_lunga = int((wr - min_wr) / range_wr * 30)
                            barra = "█" * barra_lunga
                            print(f"    {ts:>9,d} steps:  {barra}  {wr:.1f}%")
                        print()

                except Exception as e:
                    print(f"  ❌ Errore valutando {nome_file}: {type(e).__name__}: {e}")
                    gia_visti.add(nome_file)  # non riprovare

            time.sleep(args.polling)

    except KeyboardInterrupt:
        print(f"\n\nMonitor fermato manualmente.")
        if progressione:
            print(f"Riassunto: {len(progressione)} checkpoint valutati")
            print(f"Win rate iniziale: {progressione[0][1]:.1f}%")
            print(f"Win rate finale:   {progressione[-1][1]:.1f}%")
            print(f"Δ:                 {progressione[-1][1] - progressione[0][1]:+.1f}%")
            print(f"\nDati salvati in: {args.csv}")


if __name__ == "__main__":
    main()
