"""
confronta_partite.py — Confronto partita-per-partita fra due modelli.

Lancia gli stessi seed con 2 modelli e mostra:
- Per ogni seed: chi vince con modello A, chi vince con modello B
- Conteggio "swap" (partite vinte da uno ma non dall'altro)
- Conteggio partite identiche
- Pattern di miglioramento (B vince le stesse di A + altre, oppure vince partite "diverse")

Più informativo del solo win rate aggregato.

Uso:
    python scripts/confronta_partite.py modello_a.zip modello_b.zip
    python scripts/confronta_partite.py modello_a.zip modello_b.zip --n_partite 200
    python scripts/confronta_partite.py modello_a.zip modello_b.zip --bot_color GIALLO
"""

import argparse
import sys
import os
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env import RisikoEnv
from risiko_env.data import COLORI_GIOCATORI


def gioca_seed(model, seed: int, bot_color: str) -> dict:
    """Gioca una partita con seed dato, restituisce esito."""
    env = RisikoEnv(seed=seed, bot_color=bot_color)
    obs, info = env.reset()
    while True:
        mask = info["action_mask"]
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        if term or trunc:
            break
    return {
        "vincitore": info["vincitore"],
        "vinto": info["vincitore"] == bot_color,
        "reward": reward,
    }


def confronta(modello_a: str, modello_b: str, n_partite: int = 100,
              bot_color: str = "BLU", verbose_partite: int = 20) -> None:
    """Confronto partita-per-partita."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        raise ImportError("Installa: pip install sb3-contrib")

    from _helpers import carica_modello_con_autodetect
    print(f"\nCarico modello A: {modello_a}")
    model_a = carica_modello_con_autodetect(modello_a)
    print(f"Carico modello B: {modello_b}")
    model_b = carica_modello_con_autodetect(modello_b)
    # Nota: se i 2 modelli hanno dimensioni observation diverse, l'ultimo
    # caricato decide. Per confronti puliti, usa sempre modelli di stessa generazione.

    label_a = os.path.basename(modello_a).replace(".zip", "")
    label_b = os.path.basename(modello_b).replace(".zip", "")

    print(f"\nConfronto su {n_partite} partite, bot_color={bot_color}\n")

    risultati_a = []
    risultati_b = []

    # Header
    print(f"{'Seed':<6} {label_a:<25} {label_b:<25} {'esito':<15}")
    print("-" * 80)

    for seed in range(n_partite):
        ra = gioca_seed(model_a, seed, bot_color)
        rb = gioca_seed(model_b, seed, bot_color)
        risultati_a.append(ra)
        risultati_b.append(rb)

        # Verbose: mostra solo le prime N
        if seed < verbose_partite:
            sym_a = "✅" if ra["vinto"] else "❌"
            sym_b = "✅" if rb["vinto"] else "❌"
            esito = ""
            if ra["vinto"] and not rb["vinto"]:
                esito = "← A meglio"
            elif rb["vinto"] and not ra["vinto"]:
                esito = "→ B meglio"
            elif ra["vinto"] and rb["vinto"]:
                esito = "= entrambi vinti"
            else:
                esito = "= entrambi persi"
            v_a = ra["vincitore"]
            v_b = rb["vincitore"]
            print(f"{seed:<6} {sym_a} {v_a:<22} {sym_b} {v_b:<22} {esito}")

    if n_partite > verbose_partite:
        print(f"... ({n_partite - verbose_partite} partite ulteriori)")

    # Aggregazioni
    n_a_vinte = sum(1 for r in risultati_a if r["vinto"])
    n_b_vinte = sum(1 for r in risultati_b if r["vinto"])
    n_solo_a = sum(1 for ra, rb in zip(risultati_a, risultati_b)
                   if ra["vinto"] and not rb["vinto"])
    n_solo_b = sum(1 for ra, rb in zip(risultati_a, risultati_b)
                   if rb["vinto"] and not ra["vinto"])
    n_entrambe = sum(1 for ra, rb in zip(risultati_a, risultati_b)
                     if ra["vinto"] and rb["vinto"])
    n_nessuna = sum(1 for ra, rb in zip(risultati_a, risultati_b)
                    if not ra["vinto"] and not rb["vinto"])

    # Reward
    r_a_medio = float(np.mean([r["reward"] for r in risultati_a]))
    r_b_medio = float(np.mean([r["reward"] for r in risultati_b]))

    print()
    print("=" * 80)
    print(f"  RIASSUNTO ({n_partite} partite)")
    print("=" * 80)
    print()
    print(f"  Vinte da {label_a}:  {n_a_vinte:4d}  ({n_a_vinte/n_partite*100:.1f}%)")
    print(f"  Vinte da {label_b}:  {n_b_vinte:4d}  ({n_b_vinte/n_partite*100:.1f}%)")
    print()
    print(f"  Reward medio {label_a}: {r_a_medio:+.3f}")
    print(f"  Reward medio {label_b}: {r_b_medio:+.3f}")
    print(f"  Δ reward: {r_b_medio - r_a_medio:+.3f}")
    print()
    print(f"  Tabella di contingenza:")
    print(f"  ┌──────────────────────┬────────────────┬─────────────────┐")
    print(f"  │                      │  {label_b} vince │  {label_b} perde │")
    print(f"  ├──────────────────────┼────────────────┼─────────────────┤")
    print(f"  │  {label_a} vince         │     {n_entrambe:3d}        │     {n_solo_a:3d}         │")
    print(f"  │  {label_a} perde         │     {n_solo_b:3d}        │     {n_nessuna:3d}         │")
    print(f"  └──────────────────────┴────────────────┴─────────────────┘")
    print()

    # Diagnostica
    if n_b_vinte > n_a_vinte:
        if n_solo_b > n_solo_a:
            print(f"  📈 {label_b} è migliore: vince {n_solo_b} partite che {label_a} perdeva,")
            print(f"     ma perde {n_solo_a} partite che {label_a} vinceva.")
            print(f"     Guadagno netto: +{n_solo_b - n_solo_a} partite (su {n_partite})")
        else:
            print(f"  ⚠️  {label_b} ha più vittorie ma vince partite 'diverse' rispetto a {label_a}")
    elif n_a_vinte > n_b_vinte:
        print(f"  📉 {label_a} è migliore di {label_b}")
    else:
        print(f"  = Stesso win rate, ma vincono partite potenzialmente diverse")
        print(f"     ({label_a} vince {n_solo_a} che {label_b} perde, e viceversa {n_solo_b})")

    # Test McNemar (confronto matched-pairs, più preciso del z-test su due proporzioni)
    print()
    if n_solo_a + n_solo_b > 0:
        # Test McNemar approssimato: chi-quadro = (|n_solo_a - n_solo_b| - 1)^2 / (n_solo_a + n_solo_b)
        chi2 = (abs(n_solo_a - n_solo_b) - 1) ** 2 / (n_solo_a + n_solo_b)
        # p-value approssimato con chi-quadro 1 grado di libertà
        # Soglie: 3.84 → p<0.05, 6.63 → p<0.01, 10.83 → p<0.001
        if chi2 > 10.83:
            sig = "*** p<0.001"
        elif chi2 > 6.63:
            sig = "**  p<0.01"
        elif chi2 > 3.84:
            sig = "*   p<0.05"
        else:
            sig = "ns  (non significativo)"
        print(f"  Test McNemar: χ²={chi2:.2f}  →  {sig}")
        print(f"  (test matched-pairs: confronta n_solo_a={n_solo_a} vs n_solo_b={n_solo_b})")


def main():
    parser = argparse.ArgumentParser(
        description="Confronto partita-per-partita di 2 modelli sugli stessi seed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("modello_a", help="Path al modello A (.zip)")
    parser.add_argument("modello_b", help="Path al modello B (.zip)")
    parser.add_argument("--n_partite", type=int, default=100,
                        help="Numero partite (default 100)")
    parser.add_argument("--bot_color", default="BLU",
                        choices=COLORI_GIOCATORI, help="Colore POV")
    parser.add_argument("--verbose_partite", type=int, default=20,
                        help="Quante partite stampare in dettaglio (default 20)")
    args = parser.parse_args()

    if not os.path.exists(args.modello_a):
        print(f"ERRORE: file non trovato: {args.modello_a}")
        sys.exit(1)
    if not os.path.exists(args.modello_b):
        print(f"ERRORE: file non trovato: {args.modello_b}")
        sys.exit(1)

    confronta(args.modello_a, args.modello_b, args.n_partite,
              args.bot_color, args.verbose_partite)


if __name__ == "__main__":
    main()
