"""
valuta_bot.py — Valuta un bot trainato contro bot random.

Uso:
    python scripts/valuta_bot.py path/al/modello.zip [--n_partite 100]

Output:
    - Win rate del bot
    - Distribuzione delle posizioni finali
    - Reward medio
    - Tempo medio per partita
"""

import sys
import os
import argparse
import time
from collections import Counter

import numpy as np

# Aggiungi root del progetto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env import RisikoEnv, MaskableEnvWrapper


def carica_modello(path):
    """Carica un modello MaskablePPO. Importa solo se necessario."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        print("ERRORE: sb3-contrib non installato.")
        print("Installa con: pip install sb3-contrib")
        sys.exit(1)
    return MaskablePPO.load(path)


def valuta(model_path, n_partite=100, deterministic=True, verbose=False):
    """
    Valuta il bot facendolo giocare n_partite contro bot random.
    Restituisce un dict con statistiche.
    """
    model = carica_modello(model_path)
    print(f"Modello caricato: {model_path}")
    print(f"Partite di valutazione: {n_partite}\n")

    risultati = []
    inizio_totale = time.time()

    for seed in range(n_partite):
        env = MaskableEnvWrapper(RisikoEnv(seed=seed * 1000))
        obs, info = env.reset()
        step_count = 0
        inizio = time.time()

        while True:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            step_count += 1
            if terminated or truncated:
                break

        durata = time.time() - inizio
        risultati.append({
            "seed": seed,
            "reward": float(reward),
            "vincitore": info["vincitore"],
            "step": step_count,
            "durata": durata,
        })

        if verbose and (seed + 1) % 10 == 0:
            n_done = seed + 1
            wr = sum(1 for r in risultati if r["reward"] == 1.0) / n_done * 100
            print(f"  [{n_done}/{n_partite}] win rate finora: {wr:.1f}%")

    durata_totale = time.time() - inizio_totale

    # Statistiche
    n = len(risultati)
    rewards = [r["reward"] for r in risultati]
    posizioni = Counter()
    for r in risultati:
        if r["reward"] == 1.0:
            posizioni["1° (vittoria)"] += 1
        elif r["reward"] == 0.3:
            posizioni["2°"] += 1
        elif r["reward"] == -0.3:
            posizioni["3°"] += 1
        elif r["reward"] == -1.0:
            posizioni["4° o eliminato"] += 1
        else:
            posizioni["altro"] += 1

    return {
        "n_partite": n,
        "vittorie": posizioni.get("1° (vittoria)", 0),
        "win_rate": posizioni.get("1° (vittoria)", 0) / n,
        "posizioni": dict(posizioni),
        "reward_medio": float(np.mean(rewards)),
        "step_medi": np.mean([r["step"] for r in risultati]),
        "durata_totale": durata_totale,
    }


def stampa_risultati(stats):
    print("\n" + "=" * 60)
    print(f"RISULTATI VALUTAZIONE")
    print("=" * 60)
    print(f"Partite giocate:      {stats['n_partite']}")
    print(f"Win rate:             {stats['win_rate']*100:.1f}% ({stats['vittorie']}/{stats['n_partite']})")
    print(f"Reward medio:         {stats['reward_medio']:+.3f}")
    print(f"Step medi/partita:    {stats['step_medi']:.0f}")
    print(f"Durata totale:        {stats['durata_totale']:.1f}s")
    print(f"\nDistribuzione posizioni:")
    for pos, count in stats["posizioni"].items():
        pct = count / stats["n_partite"] * 100
        print(f"  {pos:20}: {count:4} ({pct:5.1f}%)")

    # Confronto con baseline
    print(f"\nBaseline:")
    print(f"  Bot random:           ~25% win rate (caso)")
    print(f"  Bot ben addestrato:   60-80% win rate")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Valuta un bot RisiKo trainato")
    parser.add_argument("model_path", help="Path al file .zip del modello")
    parser.add_argument("--n_partite", type=int, default=100,
                       help="Numero di partite di valutazione (default: 100)")
    parser.add_argument("--stocastico", action="store_true",
                       help="Usa policy stocastica invece di deterministica")
    parser.add_argument("--verbose", action="store_true",
                       help="Stampa progressi ogni 10 partite")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERRORE: file non trovato: {args.model_path}")
        sys.exit(1)

    stats = valuta(
        args.model_path,
        n_partite=args.n_partite,
        deterministic=not args.stocastico,
        verbose=args.verbose,
    )
    stampa_risultati(stats)


if __name__ == "__main__":
    main()
