"""
test_simulate_simmetrico.py — Smoke test per PR2 (MCTS simmetrico).

Verifica i 5 check richiesti:
  1. search_simmetrico non crasha (con MCTS+rete vera)
  2. dopo n_simulations la root ha figli (espansione avvenuta)
  3. almeno un figlio e' di colore diverso dal parent
     (= cambio turno gestito correttamente nell'albero)
  4. self-play simmetrico produce sample BLU > 0 E ROSSO > 0
     (regressione PR1)
  5. nessun assert "step() chiamato fuori turno bot"
     (= il bug originale di PR2 e' risolto)

Tutti i test usano una RETE NEURALE REALE (RisikoNet, con pesi random).
n_simulations basso (5-10) per velocita' nel test smoke.
"""

from __future__ import annotations
import os
import sys
import numpy as np

# Bootstrap sys.path per esecuzione standalone
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from risiko_env import RisikoEnv
from alphazero.network import RisikoNet, ACTION_DIM
from alphazero.selfplay import (
    Node,
    simulate_simmetrico,
    search_simmetrico,
    gioca_partita_selfplay_simmetrica,
)


def _build_envs_e_root(seed: int = 42):
    """Costruisce envs={"BLU": env_blu, "ROSSO": env_rosso} e una root da BLU."""
    env_blu = RisikoEnv(bot_color="BLU", mode_1v1=True, seed=seed)
    env_blu._skip_giro_avversari = True
    env_blu.reset(seed=seed)

    env_rosso = RisikoEnv(bot_color="ROSSO", mode_1v1=True, seed=None)
    env_rosso._skip_giro_avversari = True
    # alias per re-aliasing
    env_rosso.stato = env_blu.stato
    env_rosso.rng = env_blu.rng

    envs = {"BLU": env_blu, "ROSSO": env_rosso}

    root = Node(
        snapshot=env_blu.snapshot(),
        player_to_move="BLU",
        P=1.0,
    )
    return envs, root


def _new_net():
    """Rete neurale fresca (pesi random) in modalita' eval."""
    net = RisikoNet()
    net.eval()
    return net


# ─────────────────────────────────────────────────────────────────
#  TEST 1: simulate_simmetrico non crasha
# ─────────────────────────────────────────────────────────────────

def test_1_simulate_non_crasha():
    """Una singola simulate_simmetrico deve completarsi senza errori."""
    envs, root = _build_envs_e_root(seed=42)
    net = _new_net()

    simulate_simmetrico(root, envs, net)

    # Sanity: la root e' stata visitata almeno una volta
    assert root.N >= 1, f"root.N={root.N}, attesa almeno 1 visita"
    print(f"  -> OK: 1 simulate ha visitato root (N={root.N}, W={root.W:+.3f})")


# ─────────────────────────────────────────────────────────────────
#  TEST 2: dopo simulazioni la root ha figli
# ─────────────────────────────────────────────────────────────────

def test_2_root_ha_figli_dopo_simulazioni():
    """Dopo qualche simulate_simmetrico, root.children deve essere non vuoto."""
    envs, root = _build_envs_e_root(seed=42)
    net = _new_net()

    for _ in range(5):
        simulate_simmetrico(root, envs, net)

    assert len(root.children) > 0, (
        f"Root non ha figli dopo 5 simulate (N={root.N})"
    )
    print(f"  -> OK: {len(root.children)} figli espansi, root.N={root.N}")


# ─────────────────────────────────────────────────────────────────
#  TEST 3a: simulate_simmetrico funziona partendo da root ROSSO
# ─────────────────────────────────────────────────────────────────

def test_3a_simulate_da_root_rosso():
    """Verifica che simulate_simmetrico funzioni anche partendo da una root
    di colore ROSSO. Per arrivarci, simuliamo qualche turno random finche'
    giocatore_corrente diventa ROSSO, poi costruiamo la root da quel punto.

    Se questo passa, simulate sa gestire ENTRAMBE le polarita' (BLU root e
    ROSSO root). E' il check chiave della simmetria del codice."""
    env_blu = RisikoEnv(bot_color="BLU", mode_1v1=True, seed=7)
    env_blu._skip_giro_avversari = True
    obs, info = env_blu.reset(seed=7)

    env_rosso = RisikoEnv(bot_color="ROSSO", mode_1v1=True, seed=None)
    env_rosso._skip_giro_avversari = True
    env_rosso.stato = env_blu.stato
    env_rosso.rng = env_blu.rng

    # Fai finire il turno BLU con azioni random
    rng = np.random.default_rng(7)
    n_step = 0
    while env_blu.sotto_fase is not None and n_step < 300:
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        az = int(rng.choice(legali))
        obs, _, term, trunc, info = env_blu.step(az)
        n_step += 1
        if term or trunc:
            break

    assert not env_blu.stato.terminata, "Partita finita troppo presto"
    assert env_blu.stato.giocatore_corrente == "ROSSO", (
        f"Atteso ROSSO dopo fine turno BLU, trovato "
        f"{env_blu.stato.giocatore_corrente}"
    )

    # Switch a env_rosso (stesso pattern di gioca_partita_selfplay_simmetrica)
    env_rosso.stato = env_blu.stato
    env_rosso.rng = env_blu.rng
    env_rosso._inizia_fase_tris()

    envs = {"BLU": env_blu, "ROSSO": env_rosso}
    root = Node(
        snapshot=env_rosso.snapshot(),
        player_to_move="ROSSO",
        P=1.0,
    )

    net = _new_net()
    for _ in range(10):
        simulate_simmetrico(root, envs, net)

    assert root.N >= 10, f"root.N={root.N}"
    assert len(root.children) > 0, "Root ROSSO non ha figli"
    print(f"  -> OK: 10 simulate da root ROSSO, "
          f"root.N={root.N}, figli={len(root.children)}")


# ─────────────────────────────────────────────────────────────────
#  TEST 3b: l'albero contiene nodi di colore diverso quando il fine turno
#           e' raggiungibile in poche azioni (root vicino al fine turno)
# ─────────────────────────────────────────────────────────────────

def test_3b_albero_misto_quando_fine_turno_e_vicino():
    """Costruiamo una root in cui il fine turno e' raggiungibile in 1-2
    step (ad esempio: in fase SPOSTAMENTO, dove "skip" chiude il turno).
    Dopo qualche simulate, l'albero deve contenere nodi ROSSO."""
    env_blu = RisikoEnv(bot_color="BLU", mode_1v1=True, seed=42)
    env_blu._skip_giro_avversari = True
    obs, info = env_blu.reset(seed=42)

    env_rosso = RisikoEnv(bot_color="ROSSO", mode_1v1=True, seed=None)
    env_rosso._skip_giro_avversari = True
    env_rosso.stato = env_blu.stato
    env_rosso.rng = env_blu.rng

    # Avanza random finche' arriviamo a sotto_fase=SPOSTAMENTO o ATTACCO
    # (vicino al fine turno).
    rng = np.random.default_rng(42)
    target_fasi = ("spostamento", "attacco")
    max_iter = 200
    while env_blu.sotto_fase not in target_fasi and max_iter > 0:
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        az = int(rng.choice(legali))
        obs, _, term, trunc, info = env_blu.step(az)
        if term or trunc or env_blu.sotto_fase is None:
            break
        max_iter -= 1

    if env_blu.sotto_fase not in target_fasi:
        print(f"  (skip: non raggiunta fase target, sotto_fase={env_blu.sotto_fase})")
        return

    envs = {"BLU": env_blu, "ROSSO": env_rosso}
    root = Node(
        snapshot=env_blu.snapshot(),
        player_to_move="BLU",
        P=1.0,
    )

    net = _new_net()
    for _ in range(50):
        simulate_simmetrico(root, envs, net)

    # BFS sull'albero per trovare i colori
    colori_visti = set()
    queue = [root]
    n_nodi = 0
    while queue:
        n = queue.pop()
        n_nodi += 1
        if n.player_to_move is not None:
            colori_visti.add(n.player_to_move)
        queue.extend(n.children.values())

    print(f"  fase root: {env_blu.sotto_fase}, n_nodi: {n_nodi}, colori: {colori_visti}")
    assert "BLU" in colori_visti
    assert "ROSSO" in colori_visti, (
        f"ROSSO mancante: l'albero non sta espandendo oltre il fine turno BLU. "
        f"colori={colori_visti}"
    )
    print(f"  -> OK: albero misto (BLU+ROSSO) da root in fase {env_blu.sotto_fase}")


# ─────────────────────────────────────────────────────────────────
#  TEST 4: self-play simmetrico con MCTS+rete vera produce sample misti
# ─────────────────────────────────────────────────────────────────

def test_4_selfplay_mcts_genera_sample_da_entrambi_i_colori():
    """Una partita self-play completa con MCTS+rete reale deve raccogliere
    sample sia da BLU sia da ROSSO. Questo e' il check principale di PR2:
    senza il fix, esploderebbe l'assert al primo cambio turno."""
    net = _new_net()

    samples, stats = gioca_partita_selfplay_simmetrica(
        net=net,
        n_simulations=5,        # basso per velocita' nel smoke test
        seed=42,
        max_decisioni=300,      # taglio di sicurezza
        verbose=False,
    )

    print(f"  vincitore={stats['vincitore']}, motivo={stats['motivo_fine']}")
    print(f"  n_decisioni={stats['n_decisioni_totale']}, "
          f"BLU_smp={stats['n_samples_blu']}, ROSSO_smp={stats['n_samples_rosso']}")

    assert stats["n_samples_blu"] > 0, (
        f"PR2 fallito: 0 sample da BLU (samples={stats['n_samples']}, "
        f"dec={stats['n_decisioni_totale']})"
    )
    assert stats["n_samples_rosso"] > 0, (
        f"PR2 fallito: 0 sample da ROSSO (samples={stats['n_samples']}, "
        f"dec={stats['n_decisioni_totale']})"
    )
    print(f"  -> OK: BLU={stats['n_samples_blu']}, ROSSO={stats['n_samples_rosso']}")


# ─────────────────────────────────────────────────────────────────
#  TEST 5: nessun assert "step() chiamato fuori turno bot"
# ─────────────────────────────────────────────────────────────────

def test_5_no_assert_step_fuori_turno():
    """L'assert "step() chiamato fuori turno bot" era il bug originale di PR2.
    Lanciamo qualche simulate e verifichiamo che NON venga sollevata.

    Nota: se il bug fosse presente, gli altri test (1, 2, 3, 4) sarebbero
    gia' falliti. Questo e' un check esplicito di sanity."""
    envs, root = _build_envs_e_root(seed=99)
    net = _new_net()

    # Eseguo 30 simulate. Se l'assert esplode, e' AssertionError uncaught.
    try:
        for i in range(30):
            simulate_simmetrico(root, envs, net)
    except AssertionError as e:
        msg = str(e)
        if "fuori turno bot" in msg or "step() chiamato fuori turno" in msg:
            raise AssertionError(
                f"BUG ORIGINALE PR2 RIAFFIORATO al sim #{i}: {msg}"
            )
        else:
            # Altro tipo di assert - rilanciamo perche' e' inatteso
            raise

    print(f"  -> OK: 30 simulate senza assert step fuori turno (root.N={root.N})")


# ─────────────────────────────────────────────────────────────────
#  TEST 6 (bonus): root viene restorato al termine di simulate
# ─────────────────────────────────────────────────────────────────

def test_6_env_restorato_dopo_simulate():
    """Invariante della "regola d'oro" (commento di simulate.py): dopo
    simulate_simmetrico, l'env attivo deve essere allo stato della root."""
    envs, root = _build_envs_e_root(seed=42)
    net = _new_net()

    # Snapshot dello stato prima di simulate
    stato_pre_giocatore = envs["BLU"].stato.giocatore_corrente
    stato_pre_round = envs["BLU"].stato.round_corrente

    for _ in range(10):
        simulate_simmetrico(root, envs, net)

    # Dopo le simulate, BLU deve essere ancora il corrente (siamo tornati alla root)
    assert envs["BLU"].stato.giocatore_corrente == stato_pre_giocatore, (
        f"giocatore_corrente cambiato: {stato_pre_giocatore} -> "
        f"{envs['BLU'].stato.giocatore_corrente}"
    )
    assert envs["BLU"].stato.round_corrente == stato_pre_round, (
        f"round cambiato: {stato_pre_round} -> {envs['BLU'].stato.round_corrente}"
    )
    # Anche env_rosso deve essere allineato
    assert envs["ROSSO"].stato is envs["BLU"].stato, (
        "env_rosso.stato non e' allineato a env_blu.stato dopo simulate"
    )
    print(f"  -> OK: env restorato a root state (giocatore={stato_pre_giocatore}, "
          f"round={stato_pre_round})")


# ─────────────────────────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_1_simulate_non_crasha,
        test_2_root_ha_figli_dopo_simulazioni,
        test_3a_simulate_da_root_rosso,
        test_3b_albero_misto_quando_fine_turno_e_vicino,
        test_4_selfplay_mcts_genera_sample_da_entrambi_i_colori,
        test_5_no_assert_step_fuori_turno,
        test_6_env_restorato_dopo_simulate,
    ]

    failed = []
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append((t.__name__, f"AssertionError: {e}"))
        except Exception as e:
            print(f"  CRASH: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
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
