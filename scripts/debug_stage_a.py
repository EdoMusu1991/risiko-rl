"""
debug_stage_a.py — Diagnostica delle 12 feature di profilo avversari (Stage A).

Gioca una partita con bot random e stampa, ad intervalli regolari, i valori
delle feature di profilo per ogni avversario. Serve a verificare:
1. Le feature emettono valori sensati (non sempre 0, non sempre 1)?
2. I valori cambiano nel corso della partita?
3. C'è informazione utile (es. avversari aggressivi vs passivi sono distinguibili)?

Uso:
    python scripts/debug_stage_a.py
    python scripts/debug_stage_a.py --seed 42
    python scripts/debug_stage_a.py --bot_color VERDE --intervallo 5
"""

import argparse
import sys
import os
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers  # noqa: F401 (auto-applica fix UTF-8)

from risiko_env import encoding as _encoding
_encoding.STAGE_A_ATTIVO = True  # forza Stage A per il debug

from risiko_env import RisikoEnv
from risiko_env.encoding import _codifica_opponent_profile, FINESTRA_OPPONENT_PROFILE
from risiko_env.data import COLORI_GIOCATORI


def stampa_feature_avversari(env, intervallo: int = 10):
    """
    Stampa le 24 feature Stage A2 per i 3 avversari del bot (8 per avv).
    """
    storia = env._tracker.storia if hasattr(env, '_tracker') else None

    if storia is None:
        print("ERRORE: env non ha _tracker. Stage A potrebbe non essere attivo.")
        return

    # Calcola feature
    profile = _codifica_opponent_profile(env.stato, env.bot_color, storia)

    # 24 feature = 8 × 3 avversari
    avversari = [c for c in COLORI_GIOCATORI if c != env.bot_color]

    print(f"\nRound {env.stato.round_corrente} | POV: {env.bot_color}")
    print(f"  {'Avv':<8} {'Terr':>6} {'Arm':>6} {'Cont':>6} {'Conf':>6} {'CnfArm':>7} {'Min':>6} {'Cnq+':>6} {'Per-':>6}")
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")

    for i, avv in enumerate(avversari):
        feat = profile[i*8:(i+1)*8]
        print(f"  {avv:<8} {feat[0]:>6.2f} {feat[1]:>6.2f} {feat[2]:>6.2f} "
              f"{feat[3]:>6.2f} {feat[4]:>7.2f} {feat[5]:>6.2f} "
              f"{feat[6]:>6.2f} {feat[7]:>6.2f}")


def main():
    parser = argparse.ArgumentParser(description="Debug delle feature Stage A.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bot_color", type=str, default="BLU",
                        choices=["BLU", "ROSSO", "VERDE", "GIALLO"])
    parser.add_argument("--intervallo", type=int, default=10,
                        help="Stampa profili ogni N round (default 10)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"DEBUG STAGE A — seed={args.seed}, POV={args.bot_color}")
    print(f"Intervallo stampa: ogni {args.intervallo} round")
    print("=" * 70)

    env = RisikoEnv(seed=args.seed, bot_color=args.bot_color)
    obs, info = env.reset()

    # Verifica Stage A
    print(f"\nObservation shape: {obs.shape}")
    if obs.shape[0] != 342:
        print(f"WARN: observation non ha 342 feature, Stage A2 potrebbe non essere attivo.")
        return

    print("\nStato iniziale (round 0):")
    stampa_feature_avversari(env, intervallo=args.intervallo)

    # Distribuzione attacchi per stratificazione finale
    n_attacchi_per_avv = defaultdict(int)
    n_conquiste_per_avv = defaultdict(int)
    territori_strappati_pov = defaultdict(int)

    last_round_stampato = 0
    n_step = 0

    while True:
        # Bot fa una mossa random (per il debug non serve modello)
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        action = np.random.choice(legali)
        obs, reward, term, trunc, info = env.step(int(action))
        n_step += 1

        round_attuale = env.stato.round_corrente

        # Stampa periodicamente
        if round_attuale > last_round_stampato and round_attuale % args.intervallo == 0:
            stampa_feature_avversari(env, intervallo=args.intervallo)
            last_round_stampato = round_attuale

        if term or trunc:
            break

    # Stampa finale
    print("\n" + "=" * 70)
    print("STATO FINALE")
    print("=" * 70)
    stampa_feature_avversari(env, intervallo=args.intervallo)

    print(f"\nMotivo fine: {info.get('motivo_fine', 'sconosciuto')}")
    print(f"Vincitore: {info.get('vincitore', 'sconosciuto')}")
    print(f"Round totali: {env.stato.round_corrente}")
    print(f"Step bot: {n_step}")

    # Stratifica: cosa ha registrato il tracker per ogni avversario?
    print("\n" + "=" * 70)
    print("STORIA REGISTRATA PER OGNI AVVERSARIO (debug profondo)")
    print("=" * 70)

    storia = env._tracker.storia
    for avv, mosse in storia.items():
        if not mosse:
            print(f"\n{avv}: NESSUNA MOSSA REGISTRATA")
            continue

        n = len(mosse)
        n_attaccato = sum(1 for m in mosse if m.get('attaccato', False))
        tot_attacchi = sum(m.get('num_attacchi', 0) for m in mosse)
        tot_contro_pov = sum(m.get('attacchi_contro_pov', 0) for m in mosse)
        tot_conquiste = sum(m.get('territori_conquistati', 0) for m in mosse)

        print(f"\n{avv} ({n} mosse registrate):")
        print(f"  Turni con conquiste: {n_attaccato}/{n} ({n_attaccato/n*100:.0f}%)")
        print(f"  Tot. territori conquistati: {tot_conquiste}")
        print(f"  Tot. territori strappati al POV ({env.bot_color}): {tot_contro_pov}")

        # Distribuzione mosse non vuote
        if n_attaccato > 0:
            esempi = [m for m in mosse if m.get('attaccato', False)][:3]
            print(f"  Primi 3 turni con conquiste:")
            for m in esempi:
                print(f"    Turno {m['turno']}: num_attacchi={m['num_attacchi']}, "
                      f"contro_pov={m['attacchi_contro_pov']}, "
                      f"terr_conquistati={m['territori_conquistati']}")

    # Diagnosi qualità feature
    print("\n" + "=" * 70)
    print("DIAGNOSI QUALITA' FEATURE STAGE A2")
    print("=" * 70)
    profile = _codifica_opponent_profile(env.stato, env.bot_color, storia)
    avversari = [c for c in COLORI_GIOCATORI if c != env.bot_color]

    for i, avv in enumerate(avversari):
        feat = profile[i*8:(i+1)*8]
        terr, arm, cont, conf, cnfA, minc, cnq, per = feat

        print(f"\n{avv}:")
        print(f"  territori_norm       = {terr:.3f} ({int(terr*42)}/42)")
        print(f"  armate_norm          = {arm:.3f} ({int(arm*130)}/130)")
        print(f"  continenti_norm      = {cont:.3f} ({int(cont*6)}/6)")
        print(f"  confini_pov_norm     = {conf:.3f} ({int(conf*42)}/42 miei adiacenti)")
        print(f"  armate_confini_norm  = {cnfA:.3f} ({cnfA*100:.0f}% sue armate sui confini)")
        print(f"  miei_minacciati_norm = {minc:.3f} ({minc*100:.0f}% miei territori in pericolo)")
        print(f"  conquiste_recenti    = {cnq:.3f} (ultimi {FINESTRA_OPPONENT_PROFILE} turni)")
        print(f"  perdite_recenti      = {per:.3f} (ultimi {FINESTRA_OPPONENT_PROFILE} turni)")

    # Check generale: spread tra avversari
    print("\n" + "=" * 70)
    print("DISCRIMINANZA: differenze tra i 3 avversari")
    print("=" * 70)
    for j, nome in enumerate(['terr', 'arm', 'cont', 'conf', 'cnfArm', 'min', 'cnq', 'per']):
        valori = [profile[i*8+j] for i in range(len(avversari))]
        spread = max(valori) - min(valori)
        flag = "[OK]" if spread > 0.05 else "[POVERA]" if spread > 0.001 else "[INUTILE]"
        print(f"  {nome:<8} min={min(valori):.3f} max={max(valori):.3f} "
              f"spread={spread:.3f} {flag}")


if __name__ == "__main__":
    main()
