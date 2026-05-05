"""
test_selfplay_simmetrico.py — Smoke test per gioca_partita_selfplay_simmetrica (PR1).

Obiettivo: verificare che il pattern del self-play simmetrico funziona, cioe':
  1. La partita finisce in un numero ragionevole di decisioni.
  2. Vengono raccolti sample sia da BLU sia da ROSSO (n_samples_blu > 0
     E n_samples_rosso > 0). Questo e' l'invariante chiave del fix.
  3. value_target ha segno opposto fra BLU e ROSSO (a meno di pareggio).
  4. player_at_state in {BLU, ROSSO}.
  5. Invariante runtime: env_attivo.bot_color == giocatore_corrente sempre.
     (gia' check-ata dall'assert dentro la funzione, ma la verifichiamo
     indirettamente assicurandoci che nessuna eccezione venga sollevata).

Strategia: usiamo policy_fn=policy_random per sostituire MCTS+rete e rendere
il test veloce e indipendente da torch/network reale. La funzione vera
(con MCTS) verra' validata in integrazione successivamente.
"""

from __future__ import annotations
import sys
import os
import numpy as np

# Permette di lanciare il test direttamente con `python tests/test_selfplay_simmetrico.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from alphazero.selfplay.self_play import (
    gioca_partita_selfplay_simmetrica,
    TrainingSample,
)
from alphazero.network import ACTION_DIM


def policy_random(env, obs, info, temperature, rng):
    """
    Stub di MCTS+rete: sceglie azione random uniforme tra le legali e
    ritorna policy_target uniforme sulle legali (zero altrove).
    """
    mask = info["action_mask"]
    legali = np.where(mask)[0]
    assert len(legali) > 0, "policy_fn chiamato senza azioni legali"
    idx = int(rng.integers(0, len(legali)))
    action = int(legali[idx])

    policy_target = np.zeros(ACTION_DIM, dtype=np.float32)
    policy_target[legali] = 1.0 / len(legali)
    return action, policy_target


# ─────────────────────────────────────────────────────────────────
#  TEST CASES
# ─────────────────────────────────────────────────────────────────

def test_partita_simmetrica_genera_sample_da_entrambi_i_colori():
    """Invariante chiave PR1: BLU e ROSSO devono entrambi generare sample."""
    samples, stats = gioca_partita_selfplay_simmetrica(
        net=None,
        seed=42,
        max_decisioni=2000,
        verbose=False,
        policy_fn=policy_random,
    )

    print(f"  n_samples = {stats['n_samples']}")
    print(f"  n_samples_blu   = {stats['n_samples_blu']}")
    print(f"  n_samples_rosso = {stats['n_samples_rosso']}")
    print(f"  n_decisioni_totale = {stats['n_decisioni_totale']}")
    print(f"  vincitore   = {stats['vincitore']}")
    print(f"  motivo_fine = {stats['motivo_fine']}")
    print(f"  reward_finale = {stats['reward_finale']:+.3f} (POV {stats['ultimo_player']})")

    # Check 1: la partita e' finita (terminata o truncata; se truncated e'
    # un problema diverso ma non quello che testiamo qui)
    assert stats["n_decisioni_totale"] > 0, "Nessuna decisione presa"
    assert stats["partita_terminata"] or stats["truncated"], (
        "Partita non terminata e non troncata"
    )

    # Check 2: l'INVARIANTE PR1
    assert stats["n_samples_blu"] > 0, (
        f"Bug simmetria NON corretto: 0 sample da BLU "
        f"(samples={stats['n_samples']})"
    )
    assert stats["n_samples_rosso"] > 0, (
        f"Bug simmetria NON corretto: 0 sample da ROSSO "
        f"(samples={stats['n_samples']})"
    )

    # Check 3: tutti i player_at_state sono BLU o ROSSO
    colori_visti = set(s.player_at_state for s in samples)
    assert colori_visti.issubset({"BLU", "ROSSO"}), (
        f"Trovati player_at_state inattesi: {colori_visti}"
    )

    # Check 4: somma BLU + ROSSO = n_samples (no orfani)
    assert stats["n_samples_blu"] + stats["n_samples_rosso"] == stats["n_samples"]

    print("  -> OK: BLU e ROSSO entrambi rappresentati")


def test_value_target_vincitore_perdente():
    """value_target deve riflettere correttamente il vincitore:
    - vincitore: value_target = +1.0
    - perdente:  value_target = -1.0
    - pareggio:  entrambi 0.0
    Questo check e' piu' rigoroso di "segni opposti": cattura il bug
    in cui il vincitore avesse value_target sbagliato di segno."""
    samples, stats = gioca_partita_selfplay_simmetrica(
        net=None,
        seed=42,
        max_decisioni=2000,
        verbose=False,
        policy_fn=policy_random,
    )

    vincitore = stats["vincitore"]
    if not stats["partita_terminata"]:
        print("  (partita troncata, skip)")
        return
    if not samples:
        print("  (nessun sample, skip)")
        return

    valori_blu = [s.value_target for s in samples if s.player_at_state == "BLU"]
    valori_rosso = [s.value_target for s in samples if s.player_at_state == "ROSSO"]

    # Tutti i sample dello stesso colore: stesso value_target
    if valori_blu:
        v_blu = valori_blu[0]
        assert all(v == v_blu for v in valori_blu), (
            f"value_target BLU non uniforme: {set(valori_blu)}"
        )
    if valori_rosso:
        v_rosso = valori_rosso[0]
        assert all(v == v_rosso for v in valori_rosso), (
            f"value_target ROSSO non uniforme: {set(valori_rosso)}"
        )

    print(f"  vincitore={vincitore}, v_blu={valori_blu[0]:+.1f}, v_rosso={valori_rosso[0]:+.1f}")

    if vincitore == "BLU":
        assert valori_blu and valori_blu[0] == 1.0, (
            f"BLU vincitore ma value_target={valori_blu}"
        )
        assert valori_rosso and valori_rosso[0] == -1.0, (
            f"ROSSO perdente ma value_target={valori_rosso}"
        )
    elif vincitore == "ROSSO":
        assert valori_rosso and valori_rosso[0] == 1.0, (
            f"ROSSO vincitore ma value_target={valori_rosso}"
        )
        assert valori_blu and valori_blu[0] == -1.0, (
            f"BLU perdente ma value_target={valori_blu}"
        )
    else:
        # Pareggio (es. cap_sicurezza con punteggi pari)
        assert valori_blu and valori_blu[0] == 0.0
        assert valori_rosso and valori_rosso[0] == 0.0

    print("  -> OK: value_target coerenti col vincitore")


def test_seed_riproducibile():
    """Stesso seed = stessa partita (importante per debug)."""
    s1, st1 = gioca_partita_selfplay_simmetrica(
        net=None, seed=123, policy_fn=policy_random, max_decisioni=500,
    )
    s2, st2 = gioca_partita_selfplay_simmetrica(
        net=None, seed=123, policy_fn=policy_random, max_decisioni=500,
    )

    assert st1["n_decisioni_totale"] == st2["n_decisioni_totale"], (
        f"Decisioni diverse: {st1['n_decisioni_totale']} vs {st2['n_decisioni_totale']}"
    )
    assert st1["vincitore"] == st2["vincitore"]
    assert st1["n_samples_blu"] == st2["n_samples_blu"]
    assert st1["n_samples_rosso"] == st2["n_samples_rosso"]
    print(f"  -> OK: seed=123 riproducibile ({st1['n_decisioni_totale']} decisioni)")


def test_seed_diversi_partite_diverse():
    """Seed diversi devono generare partite diverse (sanity check randomness)."""
    _, st_a = gioca_partita_selfplay_simmetrica(
        net=None, seed=1, policy_fn=policy_random, max_decisioni=500,
    )
    _, st_b = gioca_partita_selfplay_simmetrica(
        net=None, seed=2, policy_fn=policy_random, max_decisioni=500,
    )
    # Almeno qualcosa deve essere diverso (n_decisioni o vincitore)
    diversi = (
        st_a["n_decisioni_totale"] != st_b["n_decisioni_totale"]
        or st_a["vincitore"] != st_b["vincitore"]
        or st_a["n_samples_blu"] != st_b["n_samples_blu"]
    )
    assert diversi, "seed=1 e seed=2 hanno prodotto partite identiche (sospetto)"
    print(f"  -> OK: seed=1 e seed=2 differiscono")


def test_struttura_sample():
    """Verifica che ogni sample abbia campi coerenti."""
    samples, _ = gioca_partita_selfplay_simmetrica(
        net=None, seed=7, policy_fn=policy_random, max_decisioni=500,
    )

    assert len(samples) > 0
    for s in samples:
        assert isinstance(s, TrainingSample)
        assert s.obs.dtype == np.float32 or s.obs.dtype == np.float64
        assert s.mask.shape[0] == ACTION_DIM
        assert s.policy_target.shape[0] == ACTION_DIM
        # policy_target deve essere una distribuzione valida
        assert np.isclose(s.policy_target.sum(), 1.0), (
            f"policy_target non normalizzato: somma={s.policy_target.sum()}"
        )
        # policy_target deve essere zero dove la mask e' zero
        non_legali = np.where(s.mask == 0)[0]
        assert np.all(s.policy_target[non_legali] == 0), (
            "policy_target ha massa su azioni illegali"
        )
        assert s.player_at_state in ("BLU", "ROSSO")
    print(f"  -> OK: {len(samples)} sample tutti ben strutturati")


def test_verbose_non_crasha():
    """Smoke check: verbose=True non rompe il flusso."""
    _, _ = gioca_partita_selfplay_simmetrica(
        net=None, seed=99, policy_fn=policy_random, max_decisioni=200,
        verbose=True,
    )
    print("  -> OK: verbose mode funziona")


# ─────────────────────────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_partita_simmetrica_genera_sample_da_entrambi_i_colori,
        test_value_target_vincitore_perdente,
        test_seed_riproducibile,
        test_seed_diversi_partite_diverse,
        test_struttura_sample,
        test_verbose_non_crasha,
    ]

    failed = []
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append((t.__name__, str(e)))
        except Exception as e:
            print(f"  CRASH: {type(e).__name__}: {e}")
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)}")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
