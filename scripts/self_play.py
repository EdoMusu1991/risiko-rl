"""
self_play.py — Orchestratore self-play minimo (Stage D semplificato).

Loop di generazioni: ogni generazione il bot e' trainato contro un mix di
versioni precedenti di se stesso (campionate dalla "population/").

Uso:
    # Crea population/ inizializzando da un baseline
    python scripts/self_play.py init --baseline path/to/baseline.zip

    # Lancia una generazione di self-play
    python scripts/self_play.py train --gen 1 --steps 1000000

    # Valuta una generazione
    python scripts/self_play.py eval --gen 1 --n_partite 200

    # Tournament fra tutte le generazioni
    python scripts/self_play.py tournament --n_partite 100

NB: questo script e' progettato per Colab/locale con Maskable PPO.
"""

import argparse
import sys
import os
import shutil
import random as _random
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: F401

from risiko_env import encoding as _encoding
from risiko_env import RisikoEnv
from risiko_env.data import COLORI_GIOCATORI


POPULATION_DIR = "population"


def cmd_init(args):
    """Inizializza la cartella population/ copiando il baseline come gen0."""
    Path(POPULATION_DIR).mkdir(exist_ok=True)
    target = Path(POPULATION_DIR) / "gen0.zip"
    if target.exists() and not args.force:
        print(f"[ERROR] {target} esiste gia'. Usa --force per sovrascrivere.")
        sys.exit(1)
    shutil.copy(args.baseline, target)
    print(f"[OK] Inizializzato gen0 da {args.baseline}")
    print(f"     Salvato in {target}")
    print(f"\nProssimo passo: python scripts/self_play.py train --gen 1")


def lista_generazioni() -> list:
    """Restituisce i path delle generazioni esistenti, ordinati per numero."""
    if not Path(POPULATION_DIR).exists():
        return []
    files = sorted(Path(POPULATION_DIR).glob("gen*.zip"))
    return [str(f) for f in files]


def campiona_avversari(generazioni: list, n: int = 3, rng=None) -> list:
    """
    Campiona N avversari dalla population.
    Strategia: 50% gen piu' recenti (challenge), 50% mix random (diversita').
    """
    if rng is None:
        rng = _random.Random()

    if len(generazioni) == 0:
        return [None] * n  # tutti random

    if len(generazioni) <= 2:
        # Pochi modelli, campiona uniformemente
        return [rng.choice(generazioni) for _ in range(n)]

    # Prefer recenti (ultimi 3) ma con un mix
    recenti = generazioni[-3:]
    avversari = []
    for _ in range(n):
        if rng.random() < 0.6:
            avversari.append(rng.choice(recenti))
        else:
            avversari.append(rng.choice(generazioni))
    return avversari


def cmd_eval(args):
    """Valuta una generazione contro un set di avversari."""
    from _helpers import carica_modello_con_autodetect

    gen_path = Path(POPULATION_DIR) / f"gen{args.gen}.zip"
    if not gen_path.exists():
        print(f"[ERROR] {gen_path} non esiste")
        sys.exit(1)

    modello = carica_modello_con_autodetect(str(gen_path), verbose=True)
    print(f"\nModello caricato: gen{args.gen}")

    # Decide vs cosa valutare
    if args.vs == "random":
        avversari_modelli = [None, None, None]
        nome_avv = "3x random"
    elif args.vs == "gen0":
        gen0 = Path(POPULATION_DIR) / "gen0.zip"
        if not gen0.exists():
            print(f"[ERROR] gen0.zip non esiste")
            sys.exit(1)
        m0 = carica_modello_con_autodetect(str(gen0), verbose=False)
        avversari_modelli = [m0, m0, m0]
        nome_avv = "3x gen0"
    elif args.vs.startswith("gen"):
        # Es: --vs gen2
        n = int(args.vs[3:])
        target = Path(POPULATION_DIR) / f"gen{n}.zip"
        if not target.exists():
            print(f"[ERROR] {target} non esiste")
            sys.exit(1)
        m = carica_modello_con_autodetect(str(target), verbose=False)
        avversari_modelli = [m, m, m]
        nome_avv = f"3x gen{n}"
    else:
        print(f"[ERROR] --vs invalido: {args.vs}. Usa 'random', 'gen0', o 'genN'")
        sys.exit(1)

    print(f"Avversari: {nome_avv}")
    print(f"Partite: {args.n_partite}")
    print()

    # Pre-imposta Stage A per il modello principale
    dim = modello.observation_space.shape[0]
    if dim == 318:
        _encoding.STAGE_A_ATTIVO = False
    elif dim in (330, 342):
        _encoding.STAGE_A_ATTIVO = True

    n_vinte = 0
    rewards = []
    posizioni = Counter()
    eliminati = 0

    for seed in range(args.n_partite):
        avv_dict = {}
        cols_avv = [c for c in COLORI_GIOCATORI if c != "BLU"]
        for i, c in enumerate(cols_avv):
            avv_dict[c] = avversari_modelli[i]

        env = RisikoEnv(seed=seed, avversari=avv_dict)
        obs, info = env.reset()
        while True:
            mask = info["action_mask"]
            action, _ = modello.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            if term or trunc:
                break

        if reward == 1.0:
            n_vinte += 1
            posizioni[1] += 1
        elif reward == 0.3:
            posizioni[2] += 1
        elif reward == -0.3:
            posizioni[3] += 1
        else:
            posizioni[4] += 1
        rewards.append(reward)
        if not env.stato.giocatori["BLU"].vivo:
            eliminati += 1

        if (seed + 1) % 50 == 0:
            print(f"  ...{seed+1}/{args.n_partite} partite (WR finora: {n_vinte/(seed+1)*100:.1f}%)")

    wr = n_vinte / args.n_partite
    print(f"\n=== RISULTATO ===")
    print(f"  Win rate:    {wr*100:.1f}% ({n_vinte}/{args.n_partite})")
    print(f"  Reward medio: {np.mean(rewards):+.3f}")
    print(f"  Eliminato:   {eliminati/args.n_partite*100:.1f}%")
    print(f"  Posizioni:   1°={posizioni[1]} 2°={posizioni[2]} 3°={posizioni[3]} 4°={posizioni[4]}")


def cmd_tournament(args):
    """Round-robin fra tutte le generazioni."""
    from _helpers import carica_modello_con_autodetect

    generazioni = lista_generazioni()
    if len(generazioni) < 2:
        print(f"[ERROR] Servono almeno 2 generazioni. Trovate: {len(generazioni)}")
        sys.exit(1)

    print(f"Tournament fra {len(generazioni)} generazioni")
    print(f"Partite per match: {args.n_partite}")
    print()

    # Tabella WR: row vs col
    wr_table = {}

    for path_a in generazioni:
        nome_a = Path(path_a).stem
        wr_table[nome_a] = {}
        modello_a = carica_modello_con_autodetect(path_a, verbose=False)

        for path_b in generazioni:
            nome_b = Path(path_b).stem
            if nome_a == nome_b:
                wr_table[nome_a][nome_b] = "—"
                continue

            modello_b = carica_modello_con_autodetect(path_b, verbose=False)

            # A vs 3x B
            n_vinte = 0
            for seed in range(args.n_partite):
                env = RisikoEnv(
                    seed=seed,
                    avversari={c: modello_b for c in COLORI_GIOCATORI if c != "BLU"}
                )
                obs, info = env.reset()
                # Setup encoding per A
                dim_a = modello_a.observation_space.shape[0]
                if dim_a == 318:
                    _encoding.STAGE_A_ATTIVO = False
                elif dim_a in (330, 342):
                    _encoding.STAGE_A_ATTIVO = True

                while True:
                    mask = info["action_mask"]
                    action, _ = modello_a.predict(obs, action_masks=mask, deterministic=True)
                    obs, reward, term, trunc, info = env.step(int(action))
                    if term or trunc:
                        break
                if reward == 1.0:
                    n_vinte += 1
            wr = n_vinte / args.n_partite
            wr_table[nome_a][nome_b] = wr
            print(f"  {nome_a} vs 3x{nome_b}: {wr*100:.1f}%")

    # Stampa tabella
    print("\n" + "=" * 70)
    print("TABELLA TORNEO (riga vs 3x colonna)")
    print("=" * 70)
    nomi = [Path(p).stem for p in generazioni]
    print(f"{'':>10}", end=" ")
    for n in nomi:
        print(f"{n:>10}", end=" ")
    print()
    for n_row in nomi:
        print(f"{n_row:>10}", end=" ")
        for n_col in nomi:
            v = wr_table[n_row][n_col]
            if v == "—":
                print(f"{'—':>10}", end=" ")
            else:
                print(f"{v*100:>9.1f}%", end=" ")
        print()


def main():
    parser = argparse.ArgumentParser(description="Self-play orchestratore.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Inizializza population/ da baseline")
    p_init.add_argument("--baseline", required=True)
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_eval = sub.add_parser("eval", help="Valuta una generazione")
    p_eval.add_argument("--gen", type=int, required=True)
    p_eval.add_argument("--vs", default="random",
                        help="'random', 'gen0', 'genN' (default: random)")
    p_eval.add_argument("--n_partite", type=int, default=200)
    p_eval.set_defaults(func=cmd_eval)

    p_tour = sub.add_parser("tournament", help="Round-robin tournament")
    p_tour.add_argument("--n_partite", type=int, default=50)
    p_tour.set_defaults(func=cmd_tournament)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
