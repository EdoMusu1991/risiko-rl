"""
test_parallel.py — Smoke test per PR3 (self-play parallelizzato).

Verifica:
1. test_smoke_2_partite_2_worker: caso minimo, niente crash, results.length == 2.
2. test_no_crash_4_worker: 4 partite con 4 worker, smoke esteso.
3. test_clamp_n_worker: n_worker > n_partite viene clampato.
4. test_determinismo: 2 chiamate con stessi seed danno stesse stats.
5. test_parita_con_sequenziale: stats parallele matchano quelle sequenziali
   (con torch.set_num_threads(1) anche nel sequenziale per fairness).
6. test_validazione_args: errori di uso comune sono catturati con ValueError.

Tutti i test usano una rete RisikoNet vera (pesi random) e n_simulations
piccolo per velocita'. max_decisioni basso (~50): partite quasi sicuramente
truncated, ma cio' non importa per la correttezza del flusso parallelo.
"""

from __future__ import annotations
import os
import sys

# Bootstrap sys.path per esecuzione standalone
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import pytest

from alphazero.network import RisikoNet
from alphazero.selfplay import (
    gioca_n_partite_parallele,
    gioca_n_partite_vs_random_parallele,
    gioca_n_partite_match_parallele,
)
from alphazero.selfplay.self_play import gioca_partita_selfplay_simmetrica


# Parametri "smoke": piccoli per girare in pochi secondi.
SMOKE_KWARGS = dict(
    n_simulations=3,
    max_decisioni=40,
    c_puct=1.5,
)


def _new_net(seed: int = 0) -> RisikoNet:
    """Crea una rete con pesi deterministici (per riproducibilita')."""
    torch.manual_seed(seed)
    net = RisikoNet()
    net.eval()
    return net


# ─────────────────────────────────────────────────────────────────
#  TEST 1: smoke base
# ─────────────────────────────────────────────────────────────────

def test_smoke_2_partite_2_worker():
    """2 partite, 2 worker: niente crash, results lunghezza giusta, samples > 0."""
    net = _new_net()

    samples_list, stats_list = gioca_n_partite_parallele(
        net=net,
        n_partite=2,
        n_worker=2,
        base_seed=42,
        **SMOKE_KWARGS,
    )

    assert len(samples_list) == 2, f"Atteso 2 partite, ricevute {len(samples_list)}"
    assert len(stats_list) == 2

    for i, (samples, stats) in enumerate(zip(samples_list, stats_list)):
        assert len(samples) > 0, f"Partita {i}: 0 sample (atteso >0)"
        # In self-play simmetrico almeno uno dei due colori deve avere sample
        assert stats["n_samples_blu"] + stats["n_samples_rosso"] > 0, (
            f"Partita {i}: n_samples_blu+rosso == 0"
        )

    print("[test_smoke_2_partite_2_worker] OK")


# ─────────────────────────────────────────────────────────────────
#  TEST 2: concurrency piu' alta
# ─────────────────────────────────────────────────────────────────

def test_no_crash_4_worker():
    """4 partite, 4 worker: smoke esteso, niente race condition visibili."""
    net = _new_net()

    samples_list, stats_list = gioca_n_partite_parallele(
        net=net,
        n_partite=4,
        n_worker=4,
        base_seed=100,
        **SMOKE_KWARGS,
    )

    assert len(samples_list) == 4
    assert len(stats_list) == 4
    for i in range(4):
        assert len(samples_list[i]) > 0, f"Partita {i}: 0 sample"

    print("[test_no_crash_4_worker] OK")


# ─────────────────────────────────────────────────────────────────
#  TEST 3: clamp di n_worker
# ─────────────────────────────────────────────────────────────────

def test_clamp_n_worker():
    """n_worker > n_partite: il pool clampa a n_partite, niente errori."""
    net = _new_net()

    # 2 partite ma chiediamo 8 worker → deve clampare a 2
    samples_list, stats_list = gioca_n_partite_parallele(
        net=net,
        n_partite=2,
        n_worker=8,
        base_seed=0,
        **SMOKE_KWARGS,
    )
    assert len(samples_list) == 2, "Clamp non rispettato in output count"

    print("[test_clamp_n_worker] OK")


# ─────────────────────────────────────────────────────────────────
#  TEST 4: determinismo (stesso seed → stesse stats)
# ─────────────────────────────────────────────────────────────────

def test_determinismo():
    """
    Due chiamate identiche di gioca_n_partite_parallele devono produrre
    le stesse statistiche per ogni partita. La rete e' fissa, i seed sono
    fissi, e ogni worker forza torch.set_num_threads(1), quindi la
    traiettoria e' deterministica.
    """
    net = _new_net(seed=123)

    sl1, st1 = gioca_n_partite_parallele(
        net=net, n_partite=3, n_worker=3, base_seed=7, **SMOKE_KWARGS
    )
    sl2, st2 = gioca_n_partite_parallele(
        net=net, n_partite=3, n_worker=3, base_seed=7, **SMOKE_KWARGS
    )

    for i in range(3):
        # Confronto su statistiche aggregate (robusto, non bit-by-bit)
        for k in ("n_decisioni_totale", "n_decisioni_mcts", "n_samples",
                  "n_samples_blu", "n_samples_rosso", "vincitore", "truncated"):
            assert st1[i][k] == st2[i][k], (
                f"Determinismo violato partita {i} key '{k}': "
                f"run1={st1[i][k]} vs run2={st2[i][k]}"
            )

    print("[test_determinismo] OK")


# ─────────────────────────────────────────────────────────────────
#  TEST 5: parita' con sequenziale
# ─────────────────────────────────────────────────────────────────

def test_parita_con_sequenziale():
    """
    Le partite giocate in parallelo (n_worker=2) devono produrre le stesse
    statistiche delle partite giocate sequenzialmente, a parita' di seed e
    rete. Forziamo torch.set_num_threads(1) anche nel sequenziale per
    avere lo stesso ambiente BLAS dei worker (che pure forzano 1 thread).
    """
    net = _new_net(seed=7)
    base_seed = 50
    n_partite = 2

    # Sequenziale
    old_n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        seq_stats = []
        for i in range(n_partite):
            _, stats = gioca_partita_selfplay_simmetrica(
                net=net, seed=base_seed + i, **SMOKE_KWARGS,
            )
            seq_stats.append(stats)
    finally:
        torch.set_num_threads(old_n)

    # Parallelo
    _, par_stats = gioca_n_partite_parallele(
        net=net, n_partite=n_partite, n_worker=2,
        base_seed=base_seed, **SMOKE_KWARGS,
    )

    for i in range(n_partite):
        for k in ("n_decisioni_totale", "n_decisioni_mcts",
                  "n_samples_blu", "n_samples_rosso", "vincitore"):
            assert seq_stats[i][k] == par_stats[i][k], (
                f"Parita' violata partita {i} key '{k}': "
                f"seq={seq_stats[i][k]} vs par={par_stats[i][k]}"
            )

    print("[test_parita_con_sequenziale] OK")


# ─────────────────────────────────────────────────────────────────
#  TEST 6: validazione argomenti
# ─────────────────────────────────────────────────────────────────

def test_validazione_args():
    """Errori di uso comune devono produrre ValueError chiari."""
    net = _new_net()

    with pytest.raises(ValueError, match="n_partite"):
        gioca_n_partite_parallele(net, n_partite=0, n_worker=2, base_seed=0)

    with pytest.raises(ValueError, match="n_worker"):
        gioca_n_partite_parallele(net, n_partite=2, n_worker=0, base_seed=0)

    with pytest.raises(ValueError, match="seed"):
        gioca_n_partite_parallele(net, n_partite=2, n_worker=2, base_seed=0,
                                   seed=42)  # seed non va passato qui

    print("[test_validazione_args] OK")


# ─────────────────────────────────────────────────────────────────
#  TEST 7: eval vs random parallelo
# ─────────────────────────────────────────────────────────────────

def test_eval_vs_random_smoke():
    """
    Smoke test per gioca_n_partite_vs_random_parallele:
    2 partite vs random in parallelo, niente crash, statistiche coerenti.
    """
    net = _new_net()

    stats_list = gioca_n_partite_vs_random_parallele(
        net=net, n_partite=2, n_worker=2, base_seed=9000,
        bot_color="BLU",
        n_simulations=3, max_decisioni=40, c_puct=1.5,
    )

    assert len(stats_list) == 2, f"Atteso 2 risultati, ricevuti {len(stats_list)}"
    for i, st in enumerate(stats_list):
        # Ogni stats deve avere almeno la chiave vincitore
        assert "vincitore" in st, f"Partita {i}: stats senza 'vincitore'"
        # vincitore puo' essere None (truncated) o BLU/ROSSO
        assert st["vincitore"] in (None, "BLU", "ROSSO"), (
            f"Partita {i}: vincitore inaspettato {st['vincitore']!r}"
        )

    print("[test_eval_vs_random_smoke] OK")


# ─────────────────────────────────────────────────────────────────
#  TEST 8: match net_blu vs net_rosso parallelo
# ─────────────────────────────────────────────────────────────────

def test_match_due_reti_smoke():
    """
    Smoke test per gioca_n_partite_match_parallele:
    2 partite con due reti, niente crash, statistiche coerenti.
    """
    net_a = _new_net(seed=1)
    net_b = _new_net(seed=2)

    stats_list = gioca_n_partite_match_parallele(
        net_blu=net_a, net_rosso=net_b,
        n_partite=2, n_worker=2, base_seed=8000,
        n_simulations=3, max_decisioni=40, c_puct=1.5,
        temperature=0.3,
    )

    assert len(stats_list) == 2, f"Atteso 2 risultati, ricevuti {len(stats_list)}"
    for i, st in enumerate(stats_list):
        assert "vincitore" in st
        assert st["vincitore"] in (None, "BLU", "ROSSO"), (
            f"Partita {i}: vincitore inaspettato {st['vincitore']!r}"
        )
        assert st["n_decisioni_totale"] > 0

    print("[test_match_due_reti_smoke] OK")


# ─────────────────────────────────────────────────────────────────
#  Runner standalone
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    tests = [
        test_smoke_2_partite_2_worker,
        test_no_crash_4_worker,
        test_clamp_n_worker,
        test_determinismo,
        test_parita_con_sequenziale,
        test_validazione_args,
        test_eval_vs_random_smoke,
        test_match_due_reti_smoke,
    ]
    print(f"\nEseguo {len(tests)} test PR3 (parallel self-play)...\n")
    for t in tests:
        t0 = time.perf_counter()
        try:
            t()
            print(f"  ({time.perf_counter()-t0:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            raise
    print(f"\nTUTTI I TEST PASSATI")