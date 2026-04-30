"""
heatmap_territori.py — Analizza dove il bot conquista più spesso.

Fa giocare un modello N partite e produce statistiche:
- Per ogni territorio: % di partite in cui finisce in mano al bot
- Per ogni continente: % di partite in cui il bot ha controllo completo
- Ordinamento per "preferenza territoriale" del bot

Capisci se il bot ha imparato strategie di continente, se preferisce certe
zone della mappa, o se gioca random.

Uso:
    python scripts/heatmap_territori.py modello.zip
    python scripts/heatmap_territori.py modello.zip --n_partite 200
    python scripts/heatmap_territori.py modello.zip --bot_color GIALLO

Senza modello (random pure, per baseline):
    python scripts/heatmap_territori.py --random --n_partite 100
"""

import argparse
import sys
import os
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env import RisikoEnv
from risiko_env.data import COLORI_GIOCATORI, CONTINENTI, TUTTI_TERRITORI


def gioca_e_raccogli(model, seed: int, bot_color: str) -> dict:
    """Gioca una partita e raccoglie statistiche territoriali finali."""
    env = RisikoEnv(seed=seed, bot_color=bot_color)
    obs, info = env.reset()
    while True:
        mask = info["action_mask"]
        if model is None:
            legali = np.where(mask)[0]
            action = np.random.choice(legali)
        else:
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        if term or trunc:
            break

    # Snapshot finale: chi possiede cosa
    territori_bot = set(env.stato.territori_di(bot_color))
    continenti_bot = {
        cont for cont, terrs in CONTINENTI.items()
        if all(env.stato.mappa[t].proprietario == bot_color for t in terrs)
    }

    return {
        "vinto": info["vincitore"] == bot_color,
        "territori_bot": territori_bot,
        "continenti_bot": continenti_bot,
        "n_territori": len(territori_bot),
        "vivo": env.stato.giocatori[bot_color].vivo,
    }


def analizza(model, n_partite: int, bot_color: str) -> dict:
    """Aggrega statistiche su N partite."""
    print(f"Analizzo {n_partite} partite, bot_color={bot_color}...")

    territori_freq = Counter()  # territorio -> n volte controllato a fine partita
    continenti_freq = Counter()  # continente -> n volte controllato per intero
    n_terr_per_partita = []
    n_cont_per_partita = []
    n_vinte = 0
    n_eliminato = 0

    for seed in range(n_partite):
        r = gioca_e_raccogli(model, seed, bot_color)
        for t in r["territori_bot"]:
            territori_freq[t] += 1
        for c in r["continenti_bot"]:
            continenti_freq[c] += 1
        n_terr_per_partita.append(r["n_territori"])
        n_cont_per_partita.append(len(r["continenti_bot"]))
        if r["vinto"]:
            n_vinte += 1
        if not r["vivo"]:
            n_eliminato += 1

        if (seed + 1) % 20 == 0:
            print(f"  {seed+1}/{n_partite}...")

    return {
        "n_partite": n_partite,
        "bot_color": bot_color,
        "territori_freq": territori_freq,
        "continenti_freq": continenti_freq,
        "n_vinte": n_vinte,
        "n_eliminato": n_eliminato,
        "media_territori": float(np.mean(n_terr_per_partita)),
        "media_continenti": float(np.mean(n_cont_per_partita)),
    }


def stampa_heatmap(stats: dict) -> None:
    """Stampa heatmap testuale dei territori e continenti."""
    n = stats["n_partite"]

    print()
    print("=" * 70)
    print(f"  HEATMAP TERRITORI - {stats['bot_color']}  ({n} partite)")
    print("=" * 70)
    print()
    print(f"  Win rate:   {stats['n_vinte']/n*100:5.1f}%  ({stats['n_vinte']}/{n})")
    print(f"  Eliminato:  {stats['n_eliminato']/n*100:5.1f}%")
    print(f"  Territori medi finali:  {stats['media_territori']:.1f}")
    print(f"  Continenti medi finali: {stats['media_continenti']:.2f}")
    print()

    # Continenti
    print(f"  {'CONTINENTI (controllo completo)':<40}")
    print(f"  {'-' * 60}")
    for cont in CONTINENTI.keys():
        freq = stats["continenti_freq"].get(cont, 0)
        pct = freq / n * 100
        barra_lunga = int(pct / 2.5)  # 0-100% → 0-40 caratteri
        barra = "█" * barra_lunga + "░" * (40 - barra_lunga)
        print(f"  {cont.upper():<22}  {barra}  {pct:5.1f}%")

    print()
    # Territori per continente
    for cont, terrs in CONTINENTI.items():
        print(f"  {cont.upper()}:")
        # Ordina per frequenza
        terrs_con_freq = sorted(
            [(t, stats["territori_freq"].get(t, 0)) for t in terrs],
            key=lambda x: -x[1],
        )
        for t, freq in terrs_con_freq:
            pct = freq / n * 100
            barra_lunga = int(pct / 4)  # 0-100% → 0-25 caratteri
            barra = "█" * barra_lunga + "░" * (25 - barra_lunga)
            print(f"    {t:<28} {barra}  {pct:5.1f}%")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Heatmap territoriale del bot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("modello", nargs="?", default=None,
                        help="Path al modello (.zip), opzionale con --random")
    parser.add_argument("--random", action="store_true",
                        help="Bot random (per baseline)")
    parser.add_argument("--n_partite", type=int, default=100,
                        help="Partite da analizzare (default 100)")
    parser.add_argument("--bot_color", default="BLU", choices=COLORI_GIOCATORI)
    args = parser.parse_args()

    if args.random:
        model = None
        print("Modalità RANDOM: bot pesca azioni a caso.")
    else:
        if not args.modello:
            print("ERRORE: serve un modello (.zip) o --random")
            sys.exit(1)
        if not os.path.exists(args.modello):
            print(f"ERRORE: file non trovato: {args.modello}")
            sys.exit(1)
        from _helpers import carica_modello_con_autodetect
        print(f"Carico modello: {args.modello}")
        model = carica_modello_con_autodetect(args.modello)

    stats = analizza(model, args.n_partite, args.bot_color)
    stampa_heatmap(stats)


if __name__ == "__main__":
    main()
