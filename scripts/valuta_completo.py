"""
valuta_completo.py — Valutazione approfondita di un bot trainato.

Differenze rispetto a valuta_bot.py:
- Misura win rate per posizione (BLU/ROSSO/VERDE/GIALLO)
- Distribuzione completa dei piazzamenti
- Statistiche di gioco (territori, armate, continenti, durata)
- Intervalli di confidenza (Wilson score) sulle metriche chiave
- Modalità confronto: 2 modelli con test di significatività
- Output console formattato + opzionale CSV

Uso base:
    python scripts/valuta_completo.py modello.zip

Solo posizione BLU (più veloce):
    python scripts/valuta_completo.py modello.zip --solo_blu --n_partite 200

Confronto fra modelli:
    python scripts/valuta_completo.py modello_v1.zip --confronta modello_v2.zip

Salva anche su CSV:
    python scripts/valuta_completo.py modello.zip --csv risultati.csv
"""

import argparse
import sys
import os
import math
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env import RisikoEnv
from risiko_env.data import COLORI_GIOCATORI, CONTINENTI


# ─────────────────────────────────────────────────────────────────────────
#  STATISTICA DI BASE
# ─────────────────────────────────────────────────────────────────────────

def wilson_ci(n_successi: int, n_totali: int, confidenza: float = 0.95) -> tuple[float, float]:
    """
    Intervallo di confidenza di Wilson per una proporzione (più accurato del normale
    quando p è vicino a 0 o 1, o quando n è piccolo).

    Restituisce (low, high) come proporzioni 0-1.
    """
    if n_totali == 0:
        return (0.0, 0.0)
    z = 1.96 if confidenza == 0.95 else 2.576  # 95% o 99%
    p = n_successi / n_totali
    denom = 1 + z**2 / n_totali
    centro = (p + z**2 / (2 * n_totali)) / denom
    raggio = z * math.sqrt((p * (1 - p) + z**2 / (4 * n_totali)) / n_totali) / denom
    return (max(0.0, centro - raggio), min(1.0, centro + raggio))


def proportion_test(n1_succ: int, n1_tot: int, n2_succ: int, n2_tot: int) -> float:
    """
    Test di significatività per differenza tra due proporzioni (z-test).
    Restituisce p-value (two-sided).
    """
    if n1_tot == 0 or n2_tot == 0:
        return 1.0
    p1 = n1_succ / n1_tot
    p2 = n2_succ / n2_tot
    p_pooled = (n1_succ + n2_succ) / (n1_tot + n2_tot)
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1_tot + 1/n2_tot))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    # Approx normale: p-value bilaterale
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return p_value


def _norm_cdf(x: float) -> float:
    """CDF della normale standard, approssimata con erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ─────────────────────────────────────────────────────────────────────────
#  VALUTAZIONE
# ─────────────────────────────────────────────────────────────────────────

def gioca_partita(model, env: RisikoEnv) -> dict:
    """Gioca una partita completa e restituisce statistiche dettagliate."""
    obs, info = env.reset()
    n_step = 0
    while True:
        mask = info["action_mask"]
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        n_step += 1
        if term or trunc:
            break

    bot_color = env.bot_color
    stato = env.stato
    vincitore = info.get("vincitore")
    motivo_fine = info.get("motivo_fine")
    round_finale = info.get("round", 0)

    # Posizione finale del bot
    posizione = _posizione_finale(stato, bot_color, vincitore)

    # Stats di gioco del bot
    n_terr_bot = stato.num_territori_di(bot_color) if stato.giocatori[bot_color].vivo else 0
    n_arm_bot = stato.num_armate_di(bot_color) if stato.giocatori[bot_color].vivo else 0
    n_continenti_bot = sum(
        1 for cont, terrs in CONTINENTI.items()
        if all(stato.mappa[t].proprietario == bot_color for t in terrs)
    )

    return {
        "bot_color": bot_color,
        "vincitore": vincitore,
        "posizione_bot": posizione,
        "vinto": vincitore == bot_color,
        "eliminato": not stato.giocatori[bot_color].vivo,
        "reward": reward,
        "n_step": n_step,
        "round_finale": round_finale,
        "motivo_fine": motivo_fine,
        "n_territori_bot": n_terr_bot,
        "n_armate_bot": n_arm_bot,
        "n_continenti_bot": n_continenti_bot,
    }


def _posizione_finale(stato, bot_color: str, vincitore: str) -> int:
    """Calcola posizione finale del bot (1=vince, 4=ultimo/eliminato)."""
    if vincitore == bot_color:
        return 1
    if not stato.giocatori[bot_color].vivo:
        return 4

    from risiko_env.obiettivi import calcola_punti_in_obiettivo
    punti = {
        c: calcola_punti_in_obiettivo(stato, c)
        for c in COLORI_GIOCATORI
        if stato.giocatori[c].vivo
    }
    ordinati = sorted(punti.items(), key=lambda x: x[1], reverse=True)
    for posizione, (c, _) in enumerate(ordinati, start=1):
        if c == bot_color:
            return posizione
    return 4


def valuta(
    model_path: str,
    n_partite: int = 100,
    posizioni: list[str] | None = None,
    seed_base: int = 0,
) -> dict:
    """
    Valuta un modello su n_partite per ogni posizione.

    Restituisce dict con statistiche complete.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        raise ImportError("Installa: pip install sb3-contrib")

    print(f"Caricamento modello: {model_path}")
    from _helpers import carica_modello_con_autodetect
    model = carica_modello_con_autodetect(model_path)

    if posizioni is None:
        posizioni = list(COLORI_GIOCATORI)

    risultati_per_pos: dict = {}

    for pos in posizioni:
        print(f"\nGioco {n_partite} partite come {pos}...")
        partite = []
        for seed in range(n_partite):
            env = RisikoEnv(seed=seed_base + seed, bot_color=pos)
            partite.append(gioca_partita(model, env))

        # Aggrega
        n_vinte = sum(1 for p in partite if p["vinto"])
        n_eliminato = sum(1 for p in partite if p["eliminato"])
        distrib_pos = Counter(p["posizione_bot"] for p in partite)
        rewards = [p["reward"] for p in partite]
        steps = [p["n_step"] for p in partite]
        rounds = [p["round_finale"] for p in partite]
        territori = [p["n_territori_bot"] for p in partite]
        armate = [p["n_armate_bot"] for p in partite]
        continenti = [p["n_continenti_bot"] for p in partite]
        motivi_fine = Counter(p["motivo_fine"] for p in partite)

        wr_low, wr_high = wilson_ci(n_vinte, n_partite)

        risultati_per_pos[pos] = {
            "n_partite": n_partite,
            "n_vinte": n_vinte,
            "win_rate": n_vinte / n_partite,
            "win_rate_ci": (wr_low, wr_high),
            "n_eliminato": n_eliminato,
            "distrib_posizioni": dict(distrib_pos),
            "reward_medio": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "step_medio": float(np.mean(steps)),
            "round_medio": float(np.mean(rounds)),
            "territori_medio": float(np.mean(territori)),
            "armate_media": float(np.mean(armate)),
            "continenti_medio": float(np.mean(continenti)),
            "motivi_fine": dict(motivi_fine),
        }

    return risultati_per_pos


# ─────────────────────────────────────────────────────────────────────────
#  STAMPA RISULTATI
# ─────────────────────────────────────────────────────────────────────────

def stampa_risultati(risultati: dict, titolo: str = "RISULTATI") -> None:
    """Stampa i risultati in formato leggibile."""
    print(f"\n{'='*70}")
    print(f"  {titolo}")
    print(f"{'='*70}\n")

    for pos, r in risultati.items():
        wr_pct = r["win_rate"] * 100
        ci_low_pct = r["win_rate_ci"][0] * 100
        ci_high_pct = r["win_rate_ci"][1] * 100
        n = r["n_partite"]

        print(f"━━━ Posizione {pos} ({n} partite) ━━━")
        print(f"  Win rate:     {wr_pct:5.1f}%  (95% CI: {ci_low_pct:.1f}% - {ci_high_pct:.1f}%)")
        print(f"  Eliminato:    {r['n_eliminato']/n*100:5.1f}%  ({r['n_eliminato']}/{n})")
        print()
        print(f"  Distribuzione posizioni:")
        for p in [1, 2, 3, 4]:
            n_pos = r["distrib_posizioni"].get(p, 0)
            barra = "█" * int(n_pos / n * 40)
            print(f"    {p}°: {n_pos/n*100:5.1f}%  {barra}")
        print()
        print(f"  Reward medio: {r['reward_medio']:+.3f}  (std: {r['reward_std']:.3f})")
        print(f"  Step medio:   {r['step_medio']:.0f}")
        print(f"  Round medio:  {r['round_medio']:.1f}")
        print()
        print(f"  Stats di gioco:")
        print(f"    Territori finali medi: {r['territori_medio']:.1f}")
        print(f"    Armate finali medie:   {r['armate_media']:.1f}")
        print(f"    Continenti controllati: {r['continenti_medio']:.2f}")
        print(f"  Motivi fine: {r['motivi_fine']}")
        print()

    # Aggregato totale
    if len(risultati) > 1:
        n_tot = sum(r["n_partite"] for r in risultati.values())
        n_vinte_tot = sum(r["n_vinte"] for r in risultati.values())
        wr_tot = n_vinte_tot / n_tot
        ci_low, ci_high = wilson_ci(n_vinte_tot, n_tot)
        print(f"━━━ AGGREGATO ({n_tot} partite totali) ━━━")
        print(f"  Win rate complessivo: {wr_tot*100:5.1f}%  (95% CI: {ci_low*100:.1f}% - {ci_high*100:.1f}%)")
        print(f"  Atteso baseline simmetrica: ~25%")
        print()


def confronta_modelli(risultati_a: dict, risultati_b: dict,
                       label_a: str, label_b: str) -> None:
    """Stampa confronto fra due modelli con test statistici."""
    print(f"\n{'='*70}")
    print(f"  CONFRONTO: {label_a}  vs  {label_b}")
    print(f"{'='*70}\n")

    print(f"{'Posizione':<10} {label_a:<20} {label_b:<20} {'Δ':<10} {'p-value':<10}")
    print("-" * 70)

    n_vinte_tot_a = 0
    n_vinte_tot_b = 0
    n_tot_a = 0
    n_tot_b = 0

    for pos in risultati_a.keys():
        if pos not in risultati_b:
            continue
        ra = risultati_a[pos]
        rb = risultati_b[pos]
        wr_a = ra["win_rate"] * 100
        wr_b = rb["win_rate"] * 100
        delta = wr_b - wr_a
        p = proportion_test(ra["n_vinte"], ra["n_partite"],
                           rb["n_vinte"], rb["n_partite"])
        sign = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{pos:<10} {wr_a:>6.1f}%               {wr_b:>6.1f}%               "
              f"{delta:+6.1f}%   p={p:.4f} {sign}")
        n_vinte_tot_a += ra["n_vinte"]
        n_vinte_tot_b += rb["n_vinte"]
        n_tot_a += ra["n_partite"]
        n_tot_b += rb["n_partite"]

    if n_tot_a > 0 and n_tot_b > 0:
        wr_tot_a = n_vinte_tot_a / n_tot_a * 100
        wr_tot_b = n_vinte_tot_b / n_tot_b * 100
        p_tot = proportion_test(n_vinte_tot_a, n_tot_a, n_vinte_tot_b, n_tot_b)
        sign_tot = "***" if p_tot < 0.001 else "**" if p_tot < 0.01 else "*" if p_tot < 0.05 else "ns"
        print("-" * 70)
        print(f"{'TOTALE':<10} {wr_tot_a:>6.1f}%               {wr_tot_b:>6.1f}%               "
              f"{wr_tot_b-wr_tot_a:+6.1f}%   p={p_tot:.4f} {sign_tot}")

    print()
    print("Legenda: *** p<0.001, ** p<0.01, * p<0.05, ns = non significativo")
    print()


def salva_csv(risultati: dict, percorso: str, modello_label: str) -> None:
    """Salva i risultati in CSV per tracking nel tempo."""
    import csv
    esiste = os.path.exists(percorso)
    with open(percorso, "a", newline="") as f:
        writer = csv.writer(f)
        if not esiste:
            writer.writerow([
                "modello", "posizione", "n_partite", "win_rate", "ci_low", "ci_high",
                "reward_medio", "step_medio", "round_medio", "territori_medio",
                "armate_media", "continenti_medio", "elim_pct",
            ])
        for pos, r in risultati.items():
            writer.writerow([
                modello_label, pos, r["n_partite"],
                f"{r['win_rate']:.4f}",
                f"{r['win_rate_ci'][0]:.4f}",
                f"{r['win_rate_ci'][1]:.4f}",
                f"{r['reward_medio']:.4f}",
                f"{r['step_medio']:.1f}",
                f"{r['round_medio']:.1f}",
                f"{r['territori_medio']:.2f}",
                f"{r['armate_media']:.2f}",
                f"{r['continenti_medio']:.3f}",
                f"{r['n_eliminato']/r['n_partite']:.4f}",
            ])
    print(f"Risultati salvati in: {percorso}")


def aggiorna_best_model(
    risultati: dict,
    modello_path: str,
    modello_label: str,
    best_path: str = "best_model.json",
) -> None:
    """
    Confronta il modello corrente con il "best model" salvato finora.
    Aggiorna best_model.json se questo è migliore.

    Metrica usata: win rate aggregato (somma vittorie / somma partite per tutte
    le posizioni testate).
    """
    import json
    n_vinte = sum(r["n_vinte"] for r in risultati.values())
    n_tot = sum(r["n_partite"] for r in risultati.values())
    win_rate = n_vinte / n_tot if n_tot > 0 else 0.0

    nuovo_record = {
        "modello": modello_label,
        "modello_path": os.path.abspath(modello_path),
        "win_rate_aggregato": win_rate,
        "n_partite_totali": n_tot,
        "win_rate_ci": list(wilson_ci(n_vinte, n_tot)),
        "win_rate_per_posizione": {
            pos: r["win_rate"] for pos, r in risultati.items()
        },
        "reward_medio_per_posizione": {
            pos: r["reward_medio"] for pos, r in risultati.items()
        },
    }

    print()
    print("=" * 70)
    print("  TRACKING BEST MODEL")
    print("=" * 70)

    if os.path.exists(best_path):
        with open(best_path) as f:
            current_best = json.load(f)
        cb_wr = current_best.get("win_rate_aggregato", 0.0)
        cb_label = current_best.get("modello", "?")
        print(f"  Best attuale: {cb_label} ({cb_wr*100:.1f}%)")
        print(f"  Candidato:    {modello_label} ({win_rate*100:.1f}%)")

        # Test di significatività: il candidato è meglio?
        cb_n_vinte = int(cb_wr * current_best.get("n_partite_totali", 0))
        cb_n_tot = current_best.get("n_partite_totali", 0)
        if cb_n_tot > 0:
            p_value = proportion_test(cb_n_vinte, cb_n_tot, n_vinte, n_tot)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  Differenza:   {(win_rate - cb_wr)*100:+.1f}%  (p={p_value:.4f} {sig})")

        if win_rate > cb_wr:
            with open(best_path, "w") as f:
                json.dump(nuovo_record, f, indent=2)
            print(f"  ✅ NUOVO BEST! Salvato in {best_path}")
        else:
            print(f"  Best non aggiornato (candidato più debole)")
    else:
        with open(best_path, "w") as f:
            json.dump(nuovo_record, f, indent=2)
        print(f"  ✅ Primo modello tracciato. Salvato in {best_path}")
    print()


# ─────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Valutazione approfondita di un bot RisiKo trainato.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("modello", help="Path al .zip del modello")
    parser.add_argument("--n_partite", type=int, default=100,
                        help="Partite per posizione (default 100)")
    parser.add_argument("--solo_blu", action="store_true",
                        help="Solo posizione BLU (più veloce)")
    parser.add_argument("--confronta", default=None,
                        help="Path a un secondo modello per confronto")
    parser.add_argument("--csv", default=None,
                        help="Salva risultati in CSV (append)")
    parser.add_argument("--track_best", default=None, nargs="?", const="best_model.json",
                        help="Confronta con best model salvato (default: best_model.json)")
    parser.add_argument("--seed_base", type=int, default=0,
                        help="Seed di partenza (per riproducibilità)")
    args = parser.parse_args()

    if not os.path.exists(args.modello):
        print(f"ERRORE: file non trovato: {args.modello}")
        sys.exit(1)

    posizioni = ["BLU"] if args.solo_blu else list(COLORI_GIOCATORI)

    label_a = os.path.basename(args.modello).replace(".zip", "")
    risultati_a = valuta(args.modello, args.n_partite, posizioni, args.seed_base)
    stampa_risultati(risultati_a, titolo=f"MODELLO: {label_a}")

    if args.csv:
        salva_csv(risultati_a, args.csv, label_a)

    if args.track_best:
        aggiorna_best_model(risultati_a, args.modello, label_a, args.track_best)

    if args.confronta:
        if not os.path.exists(args.confronta):
            print(f"ERRORE: file confronto non trovato: {args.confronta}")
            sys.exit(1)
        label_b = os.path.basename(args.confronta).replace(".zip", "")
        risultati_b = valuta(args.confronta, args.n_partite, posizioni, args.seed_base)
        stampa_risultati(risultati_b, titolo=f"MODELLO: {label_b}")
        if args.csv:
            salva_csv(risultati_b, args.csv, label_b)
        if args.track_best:
            aggiorna_best_model(risultati_b, args.confronta, label_b, args.track_best)
        confronta_modelli(risultati_a, risultati_b, label_a, label_b)


if __name__ == "__main__":
    main()
