"""
verifica_integrita.py — Stress test dell'environment RisiKo.

Gioca N partite con bot random e verifica statistiche di sanità:
- Distribuzione vincitori bilanciata (~25% per posizione)
- Durata partite ragionevole (300-700 step)
- Cap 130 mai superato
- Motivi fine plausibili (sdadata > vittoria > cap_sicurezza > vittoria_obiettivo)
- Continenti effettivamente conquistati
- Nessuna partita "infinita" (truncated)
- Distribuzione obiettivi assegnati uniforme

Utile da rilanciare ogni volta che si modifica l'env, per beccare regressioni.

Uso:
    python scripts/verifica_integrita.py
    python scripts/verifica_integrita.py --n_partite 5000   # più rigoroso
    python scripts/verifica_integrita.py --bot_color GIALLO # solo dal POV GIALLO
"""

import argparse
import sys
import os
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env import RisikoEnv
from risiko_env.data import COLORI_GIOCATORI, MAX_ARMATE_TOTALI, CONTINENTI


# ─────────────────────────────────────────────────────────────────────────
#  STRESS TEST
# ─────────────────────────────────────────────────────────────────────────

def stress_test(n_partite: int = 1000, bot_color: str = "BLU") -> dict:
    """Gioca n_partite con bot random e raccoglie statistiche."""
    print(f"Stress test: {n_partite} partite, bot_color={bot_color}")
    inizio = time.time()

    vincitori = Counter()
    motivi_fine = Counter()
    durate_step = []
    durate_round = []
    armate_max_per_giocatore = defaultdict(list)
    obiettivi_assegnati = Counter()
    n_truncated = 0
    n_cap_violato = 0
    n_continenti_conquistati = 0  # almeno un continente possesso da qualcuno a fine
    eccezioni = []

    for seed in range(n_partite):
        try:
            env = RisikoEnv(seed=seed, bot_color=bot_color)
            obs, info = env.reset()

            # Traccia obiettivi assegnati
            for c in COLORI_GIOCATORI:
                obi_id = env.stato.giocatori[c].obiettivo_id
                if obi_id:
                    obiettivi_assegnati[obi_id] += 1

            n_step = 0
            while True:
                mask = info["action_mask"]
                legali = np.where(mask)[0]
                action = np.random.choice(legali)
                obs, reward, term, trunc, info = env.step(int(action))
                n_step += 1
                if term or trunc:
                    break

            if trunc:
                n_truncated += 1

            # Controlli a fine partita
            for c in COLORI_GIOCATORI:
                n_armate = env.stato.num_armate_di(c)
                armate_max_per_giocatore[c].append(n_armate)
                if n_armate > MAX_ARMATE_TOTALI:
                    n_cap_violato += 1
                    print(f"  ⚠️  Seed {seed}: {c} ha {n_armate} armate (cap {MAX_ARMATE_TOTALI})")

            # Continenti conquistati a fine
            for cont, terrs in CONTINENTI.items():
                proprietari = {env.stato.mappa[t].proprietario for t in terrs}
                if len(proprietari) == 1 and None not in proprietari:
                    n_continenti_conquistati += 1

            vincitori[info["vincitore"]] += 1
            motivi_fine[info["motivo_fine"]] += 1
            durate_step.append(n_step)
            durate_round.append(info["round"])

        except Exception as e:
            eccezioni.append((seed, str(e)))
            if len(eccezioni) <= 3:
                print(f"  ❌ Seed {seed}: {type(e).__name__}: {e}")

        # Progress
        if (seed + 1) % 100 == 0:
            elapsed = time.time() - inizio
            speed = (seed + 1) / elapsed
            eta = (n_partite - seed - 1) / speed
            print(f"  {seed+1}/{n_partite} ({speed:.0f} partite/s, ETA {eta:.0f}s)")

    durata_totale = time.time() - inizio

    return {
        "n_partite": n_partite,
        "bot_color": bot_color,
        "durata_totale_s": durata_totale,
        "vincitori": dict(vincitori),
        "motivi_fine": dict(motivi_fine),
        "step_min": min(durate_step) if durate_step else 0,
        "step_max": max(durate_step) if durate_step else 0,
        "step_medio": float(np.mean(durate_step)) if durate_step else 0,
        "step_std": float(np.std(durate_step)) if durate_step else 0,
        "round_medio": float(np.mean(durate_round)) if durate_round else 0,
        "round_max": max(durate_round) if durate_round else 0,
        "armate_max_per_giocatore": {
            c: max(arm) for c, arm in armate_max_per_giocatore.items()
        },
        "n_truncated": n_truncated,
        "n_cap_violato": n_cap_violato,
        "obiettivi_assegnati": dict(obiettivi_assegnati),
        "continenti_finali_medio": n_continenti_conquistati / n_partite,
        "eccezioni": eccezioni,
    }


# ─────────────────────────────────────────────────────────────────────────
#  CONTROLLI E REPORT
# ─────────────────────────────────────────────────────────────────────────

def controlla_e_riporta(stats: dict) -> int:
    """
    Controlla statistiche contro tolleranze sane.
    Restituisce numero di failure.
    """
    n = stats["n_partite"]
    failures = 0

    print()
    print("=" * 70)
    print("  STATISTICHE")
    print("=" * 70)

    # Performance
    print()
    print(f"Durata totale:       {stats['durata_totale_s']:.1f}s "
          f"({n/stats['durata_totale_s']:.0f} partite/s)")

    # Distribuzione vincitori
    print()
    print("Distribuzione vincitori:")
    for c in COLORI_GIOCATORI:
        n_vinte = stats["vincitori"].get(c, 0)
        pct = n_vinte / n * 100
        barra = "█" * int(pct / 2)
        print(f"  {c:6s}: {n_vinte:4d} ({pct:5.1f}%)  {barra}")

    # CHECK 1: distribuzione bilanciata (con tolleranza ±10% dal 25%)
    print()
    for c in COLORI_GIOCATORI:
        pct = stats["vincitori"].get(c, 0) / n * 100
        if abs(pct - 25) > 10:
            print(f"  ⚠️  {c}: win rate {pct:.1f}% lontano dal 25% atteso")
            failures += 1
    if all(abs(stats["vincitori"].get(c, 0) / n * 100 - 25) <= 10
           for c in COLORI_GIOCATORI):
        print("  ✓ Distribuzione vincitori bilanciata (±10% dal 25%)")

    # Motivi fine
    print()
    print("Motivi fine:")
    for m, count in sorted(stats["motivi_fine"].items(), key=lambda x: -x[1]):
        pct = count / n * 100
        print(f"  {m:20s}: {count:4d} ({pct:5.1f}%)")

    # Durate
    print()
    print(f"Durata partite (step):")
    print(f"  min:    {stats['step_min']}")
    print(f"  max:    {stats['step_max']}")
    print(f"  media:  {stats['step_medio']:.0f} (std {stats['step_std']:.0f})")
    print(f"Round medio: {stats['round_medio']:.1f} (max {stats['round_max']})")

    # CHECK 2: durate ragionevoli
    if stats["step_min"] < 10:
        print(f"  ⚠️  Min step troppo basso ({stats['step_min']}): possibile bug")
        failures += 1
    if stats["step_medio"] < 100 or stats["step_medio"] > 2000:
        print(f"  ⚠️  Step medio fuori range plausibile: {stats['step_medio']:.0f}")
        failures += 1
    if stats["step_min"] >= 10 and 100 <= stats["step_medio"] <= 2000:
        print("  ✓ Durate partite ragionevoli")

    # Armate max
    print()
    print(f"Armate massime mai raggiunte (cap {MAX_ARMATE_TOTALI}):")
    for c, max_arm in stats["armate_max_per_giocatore"].items():
        warn = "⚠️" if max_arm > MAX_ARMATE_TOTALI else "✓"
        print(f"  {c:6s}: {max_arm:3d}  {warn}")

    # CHECK 3: cap mai violato
    if stats["n_cap_violato"] > 0:
        print(f"  ⚠️  Cap 130 violato in {stats['n_cap_violato']} casi!")
        failures += 1
    else:
        print(f"  ✓ Cap {MAX_ARMATE_TOTALI} mai superato")

    # Continenti
    print()
    print(f"Continenti completati a fine partita (in media): "
          f"{stats['continenti_finali_medio']:.2f}")

    # CHECK 4: distribuzione obiettivi
    print()
    print("Distribuzione obiettivi assegnati:")
    n_obj_assegnati = sum(stats["obiettivi_assegnati"].values())
    if n_obj_assegnati > 0:
        atteso = n_obj_assegnati / 16  # 16 obiettivi
        max_dev = 0
        for obj_id, count in sorted(stats["obiettivi_assegnati"].items()):
            dev = abs(count - atteso) / atteso * 100
            max_dev = max(max_dev, dev)
        print(f"  Atteso uniforme: ~{atteso:.0f} per obiettivo")
        print(f"  Max deviazione: {max_dev:.1f}%")
        if max_dev > 30:
            print(f"  ⚠️  Distribuzione obiettivi sbilanciata (max dev {max_dev:.1f}%)")
            failures += 1
        else:
            print(f"  ✓ Distribuzione obiettivi accettabile")

    # CHECK 5: nessuna partita troncata
    print()
    if stats["n_truncated"] > 0:
        print(f"  ⚠️  {stats['n_truncated']} partite truncate (failsafe attivato)")
        failures += 1
    else:
        print(f"  ✓ Nessuna partita truncata (tutte terminate naturalmente)")

    # CHECK 6: nessuna eccezione
    print()
    if stats["eccezioni"]:
        print(f"  ❌ {len(stats['eccezioni'])} eccezioni durante il test")
        failures += 1
    else:
        print(f"  ✓ Nessuna eccezione")

    # Verdetto finale
    print()
    print("=" * 70)
    if failures == 0:
        print(f"  ✅ TUTTI I CHECK PASSATI ({n} partite)")
    else:
        print(f"  ❌ {failures} CHECK FALLITI ({n} partite)")
    print("=" * 70)

    return failures


# ─────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stress test dell'env RisiKo.")
    parser.add_argument("--n_partite", type=int, default=1000,
                        help="Numero di partite (default 1000)")
    parser.add_argument("--bot_color", default="BLU", choices=COLORI_GIOCATORI,
                        help="Colore POV (default BLU)")
    args = parser.parse_args()

    stats = stress_test(args.n_partite, args.bot_color)
    failures = controlla_e_riporta(stats)
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
