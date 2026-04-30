"""
test_regressione.py — Test di regressione per modelli salvati.

Verifica che ogni modello in `test_models/` rispetti dei criteri minimi:
- Carica senza errori
- Win rate >= 15% (sopra il "broken" del 12% pre-fix asimmetria)
- Reward medio >= -0.5
- Nessuna eccezione durante 30 partite

Da rilanciare dopo OGNI modifica significativa al simulatore (env, encoding,
azioni, motore) per assicurarsi che modelli precedentemente trainati continuino
a funzionare.

Per usarlo:
1. Crea cartella test_models/ nella root del progetto
2. Copia dentro 1+ modelli .zip da testare (es: i tuoi best checkpoint)
3. Lancia: python tests/test_regressione.py

Se la cartella è vuota o non esiste, il test viene skippato (non fallisce).

Esegui: python tests\test_regressione.py
"""

import sys
import os
import glob
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env import RisikoEnv


# Soglie minime
SOGLIA_WIN_RATE = 0.15        # >= 15% (random simmetrico ≈ 19-25%)
SOGLIA_REWARD_MEDIO = -0.5    # bot non disastrato
N_PARTITE_TEST = 30           # veloce, abbastanza statistica


def carica_modelli_test() -> list:
    """Trova tutti i .zip in test_models/."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(root, "test_models")
    if not os.path.exists(test_dir):
        return []
    return sorted(glob.glob(os.path.join(test_dir, "*.zip")))


def test_caricamento_modelli():
    """Verifica che ogni modello si carichi senza errori."""
    modelli = carica_modelli_test()
    if not modelli:
        print("⚠ Nessun modello in test_models/, skippo")
        return

    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        print("⚠ sb3-contrib non installato, skippo")
        return

    for path in modelli:
        nome = os.path.basename(path)
        try:
            model = MaskablePPO.load(path)
            n_params = sum(p.numel() for p in model.policy.parameters())
            print(f"✓ Caricato {nome}: {n_params:,} parametri")
        except Exception as e:
            assert False, f"Errore caricando {nome}: {e}"


def test_modello_non_disastrato():
    """Verifica che il modello faccia almeno il 15% win rate."""
    modelli = carica_modelli_test()
    if not modelli:
        print("⚠ Nessun modello in test_models/, skippo")
        return

    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        print("⚠ sb3-contrib non installato, skippo")
        return

    for path in modelli:
        nome = os.path.basename(path)
        model = MaskablePPO.load(path)

        n_vinte = 0
        rewards = []
        for seed in range(N_PARTITE_TEST):
            env = RisikoEnv(seed=seed)
            obs, info = env.reset()
            while True:
                mask = info["action_mask"]
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, reward, term, trunc, info = env.step(int(action))
                if term or trunc:
                    break
            if reward == 1.0:
                n_vinte += 1
            rewards.append(reward)

        wr = n_vinte / N_PARTITE_TEST
        rm = float(np.mean(rewards))

        # CHECK 1: win rate
        assert wr >= SOGLIA_WIN_RATE, (
            f"REGRESSIONE: {nome} win rate {wr*100:.1f}% sotto soglia "
            f"{SOGLIA_WIN_RATE*100:.0f}%"
        )
        # CHECK 2: reward medio
        assert rm >= SOGLIA_REWARD_MEDIO, (
            f"REGRESSIONE: {nome} reward medio {rm:.3f} sotto soglia "
            f"{SOGLIA_REWARD_MEDIO}"
        )
        print(f"✓ {nome}: WR={wr*100:.1f}%, reward={rm:+.3f}")


def test_modello_no_eccezioni():
    """Verifica che il modello non lanci eccezioni durante le partite."""
    modelli = carica_modelli_test()
    if not modelli:
        print("⚠ Nessun modello in test_models/, skippo")
        return

    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        print("⚠ sb3-contrib non installato, skippo")
        return

    for path in modelli:
        nome = os.path.basename(path)
        model = MaskablePPO.load(path)

        for seed in range(10):  # solo 10 per velocità
            env = RisikoEnv(seed=seed)
            try:
                obs, info = env.reset()
                while True:
                    mask = info["action_mask"]
                    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                    obs, reward, term, trunc, info = env.step(int(action))
                    if term or trunc:
                        break
            except Exception as e:
                assert False, f"Eccezione su {nome} seed {seed}: {e}"
        print(f"✓ {nome}: no eccezioni in 10 partite")


# ─────────────────────────────────────────────────────────────────────────
#  TEST AGGIUNTIVI: bot random come baseline (sempre eseguibili)
# ─────────────────────────────────────────────────────────────────────────

def test_bot_random_winrate_ragionevole():
    """
    Sanity check: il bot random in qualsiasi posizione deve fare 12%-30%.
    Soglie larghe per evitare flakiness, ma stretti abbastanza da prendere
    regressioni dell'env.

    Questo test gira SEMPRE, anche senza modelli. Se fallisce, c'è un bug
    nell'env (asimmetria, distribuzione armate, ecc.).
    """
    from risiko_env.data import COLORI_GIOCATORI
    from collections import Counter

    risultati = {}
    for bot_color in COLORI_GIOCATORI:
        vincitori = Counter()
        for seed in range(50):
            env = RisikoEnv(seed=seed, bot_color=bot_color)
            obs, info = env.reset()
            while True:
                mask = info["action_mask"]
                legali = np.where(mask)[0]
                action = random.choice(legali)
                obs, reward, term, trunc, info = env.step(int(action))
                if term or trunc:
                    break
            vincitori[info["vincitore"]] += 1
        wr_pct = vincitori[bot_color] / 50 * 100
        risultati[bot_color] = wr_pct

    # Tutte le posizioni devono essere tra 8% e 35% (con N=50, varianza ~7-10%)
    # Soglie larghe: meglio non avere falsi positivi.
    for c, wr in risultati.items():
        assert 8 <= wr <= 35, (
            f"REGRESSIONE ENV: bot random come {c} fa {wr:.0f}% "
            f"(atteso 10-30% con N=50, vedi distribuzione: {risultati})"
        )
    print(f"✓ Bot random in tutte le posizioni: {risultati}")


def main():
    tests = [
        test_caricamento_modelli,
        test_modello_no_eccezioni,
        test_modello_non_disastrato,
        test_bot_random_winrate_ragionevole,
    ]

    print("\n" + "=" * 60)
    print("Test di regressione")
    print("=" * 60 + "\n")

    falliti = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} FALLITO: {e}")
            falliti.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} ERRORE: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            falliti.append(test.__name__)

    print("\n" + "=" * 60)
    if falliti:
        print(f"FALLITI: {len(falliti)}/{len(tests)}")
        for nome in falliti:
            print(f"  - {nome}")
    else:
        print(f"TUTTI I {len(tests)} TEST PASSATI ✓")
    print("=" * 60 + "\n")

    return len(falliti) == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
