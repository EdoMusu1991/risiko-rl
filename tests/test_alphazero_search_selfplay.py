"""
test_alphazero_search_selfplay.py — Test search() e gioca_partita_selfplay().

Sub-step 5 della Settimana 4. NB: questi test usano n_sim molto basso e
max_decisioni basso per non superare il timeout. Il vero training di partite
complete avverra' su GPU in Settimana 5+.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from risiko_env import encoding as e
e.STAGE_A_ATTIVO = True
from risiko_env import RisikoEnv

from alphazero.network import RisikoNet, ACTION_DIM
from alphazero.selfplay import (
    Node, search, visite_to_policy_full,
    gioca_partita_selfplay, TrainingSample
)


def make_env_in_fase_rinforzo(seed=42):
    """Helper: crea env e avanza fino a fase rinforzo (per avere molte azioni)."""
    env = RisikoEnv(seed=seed, mode_1v1=True, reward_mode='margin')
    obs, info = env.reset()
    while env.sotto_fase != "rinforzo" or info["action_mask"].sum() <= 1:
        legali = np.where(info["action_mask"])[0]
        if len(legali) == 0:
            break
        obs, _, term, trunc, info = env.step(int(legali[0]))
        if term or trunc:
            break
    return env, obs, info


# ─────────────────────────────────────────────────────────────────
#  TEST search()
# ─────────────────────────────────────────────────────────────────

def test_search_ritorna_azione_legale():
    """L'azione restituita da search() deve essere legale."""
    env, obs, info = make_env_in_fase_rinforzo(seed=42)
    legali = set(np.where(info["action_mask"])[0])
    
    root = Node(snapshot=env.snapshot(), player_to_move=env.stato.giocatore_corrente, P=1.0)
    net = RisikoNet()
    
    action, dist = search(root, env, net, n_simulations=10, temperature=1.0)
    assert action in legali, f"Azione {action} non legale"
    print("  ✓ search ritorna azione legale")


def test_search_distribuzione_valida():
    """La distribuzione restituita da search somma a 1, e' su azioni legali."""
    env, obs, info = make_env_in_fase_rinforzo(seed=42)
    
    root = Node(snapshot=env.snapshot(), player_to_move=env.stato.giocatore_corrente, P=1.0)
    net = RisikoNet()
    
    _, dist = search(root, env, net, n_simulations=10, temperature=1.0)
    
    actions, probs = zip(*dist)
    assert abs(sum(probs) - 1.0) < 1e-5, f"Sum probs = {sum(probs)}"
    
    # Tutte le azioni nella distribuzione sono legali
    legali = set(np.where(info["action_mask"])[0])
    assert all(a in legali for a in actions)
    print(f"  ✓ search produce distribuzione valida ({len(actions)} azioni)")


def test_search_temperature_zero_argmax():
    """Con T=0, l'azione e' argmax delle visite."""
    env, obs, info = make_env_in_fase_rinforzo(seed=42)
    
    root = Node(snapshot=env.snapshot(), player_to_move=env.stato.giocatore_corrente, P=1.0)
    net = RisikoNet()
    
    action, dist = search(root, env, net, n_simulations=20, temperature=0.0)
    
    # Action == azione con piu' visite
    visits = {a: child.N for a, child in root.children.items()}
    best_visited = max(visits, key=visits.get)
    assert action == best_visited, f"Action {action} != most visited {best_visited}"
    
    # Distribuzione one-hot
    probs = [p for _, p in dist]
    assert max(probs) == 1.0
    assert sum(probs) == 1.0
    print(f"  ✓ search T=0: argmax visite (action={action})")


def test_search_env_pulito():
    """search() lascia env nello stato della root."""
    env, obs, info = make_env_in_fase_rinforzo(seed=42)
    snap_before = env.snapshot()
    
    root = Node(snapshot=snap_before, player_to_move=env.stato.giocatore_corrente, P=1.0)
    net = RisikoNet()
    
    search(root, env, net, n_simulations=20)
    
    snap_after = env.snapshot()
    assert snap_after == snap_before, "search() ha sporcato l'env!"
    print("  ✓ search lascia env pulito")


# ─────────────────────────────────────────────────────────────────
#  TEST visite_to_policy_full()
# ─────────────────────────────────────────────────────────────────

def test_policy_full_somma_uno():
    """visite_to_policy_full produce vettore 1765-D che somma a 1."""
    env, _, _ = make_env_in_fase_rinforzo(seed=42)
    root = Node(snapshot=env.snapshot(), player_to_move=env.stato.giocatore_corrente, P=1.0)
    net = RisikoNet()
    search(root, env, net, n_simulations=10)
    
    policy = visite_to_policy_full(root, ACTION_DIM, temperature=1.0)
    assert policy.shape == (ACTION_DIM,)
    assert abs(policy.sum() - 1.0) < 1e-5
    print("  ✓ visite_to_policy_full: shape e somma OK")


def test_policy_full_zero_su_illegali():
    """visite_to_policy_full ha zero sulle azioni mai esplorate (illegali)."""
    env, _, info = make_env_in_fase_rinforzo(seed=42)
    root = Node(snapshot=env.snapshot(), player_to_move=env.stato.giocatore_corrente, P=1.0)
    net = RisikoNet()
    search(root, env, net, n_simulations=10)
    
    policy = visite_to_policy_full(root, ACTION_DIM, temperature=1.0)
    mask = info["action_mask"]
    
    # Sulle azioni illegali, la policy deve essere zero
    illegal_mass = policy[~mask].sum()
    assert illegal_mass < 1e-6, f"Policy mass su illegali: {illegal_mass}"
    print("  ✓ visite_to_policy_full: zero su azioni illegali")


# ─────────────────────────────────────────────────────────────────
#  TEST gioca_partita_selfplay() — short config
# ─────────────────────────────────────────────────────────────────

def test_selfplay_produce_samples():
    """Una partita short produce samples validi."""
    env = RisikoEnv(seed=42, mode_1v1=True, reward_mode='margin')
    net = RisikoNet()
    
    samples, stats = gioca_partita_selfplay(
        env, net,
        n_simulations=5,
        seed=42,
        max_decisioni=20,
    )
    
    assert len(samples) > 0
    assert stats["n_samples"] == len(samples)
    print(f"  ✓ selfplay produce {len(samples)} samples")


def test_selfplay_samples_struttura():
    """Verifica struttura di ogni sample."""
    env = RisikoEnv(seed=42, mode_1v1=True, reward_mode='margin')
    net = RisikoNet()
    
    samples, _ = gioca_partita_selfplay(
        env, net,
        n_simulations=5,
        seed=42,
        max_decisioni=20,
    )
    
    for i, s in enumerate(samples):
        # Struttura
        assert isinstance(s, TrainingSample), f"Sample {i} non e' TrainingSample"
        assert s.obs.shape == (342,), f"Sample {i}: obs shape {s.obs.shape}"
        assert s.mask.shape == (ACTION_DIM,)
        assert s.policy_target.shape == (ACTION_DIM,)
        
        # Valori
        assert 0.0 <= s.obs.min() and s.obs.max() <= 1.0  # normalizzato
        assert s.mask.sum() >= 2  # niente fast-path
        assert abs(s.policy_target.sum() - 1.0) < 1e-4  # somma a 1
        assert s.policy_target[~s.mask].sum() < 1e-6  # zero su illegali
        assert s.player_at_state in ("BLU", "ROSSO")
        assert -1.0 <= s.value_target <= 1.0
    
    print(f"  ✓ Tutti i {len(samples)} samples ben strutturati")


def test_selfplay_simmetria_value():
    """In zero-sum 1v1: avg(value BLU) + avg(value ROSSO) ≈ 0."""
    # Forziamo una partita un po' piu' lunga per arrivare a turno ROSSO
    env = RisikoEnv(seed=42, mode_1v1=True, reward_mode='margin')
    net = RisikoNet()
    
    # NB: serve una partita che arrivi al turno di ROSSO. Con n_sim=2 va piu' veloce.
    samples, stats = gioca_partita_selfplay(
        env, net,
        n_simulations=2,
        seed=42,
        max_decisioni=80,  # alto per arrivare a ROSSO
    )
    
    blu = [s.value_target for s in samples if s.player_at_state == "BLU"]
    rosso = [s.value_target for s in samples if s.player_at_state == "ROSSO"]
    
    if blu and rosso:
        # Tutti i value BLU sono uguali (= reward finale dal POV di BLU)
        # Tutti i value ROSSO sono uguali (= -reward finale)
        # Quindi avg(BLU) + avg(ROSSO) = 0
        avg_blu = np.mean(blu)
        avg_rosso = np.mean(rosso)
        assert abs(avg_blu + avg_rosso) < 1e-6, \
            f"Simmetria rotta: BLU avg={avg_blu}, ROSSO avg={avg_rosso}"
        print(f"  ✓ Simmetria zero-sum: BLU={avg_blu:+.3f}, ROSSO={avg_rosso:+.3f}")
    else:
        # Se non sono arrivato al turno di ROSSO, salto il test
        print(f"  ~ Simmetria non testata (BLU={len(blu)}, ROSSO={len(rosso)}) — partita troppo corta")


# ─────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test search() + gioca_partita_selfplay()")
    print("=" * 55)
    
    print("\nsearch():")
    test_search_ritorna_azione_legale()
    test_search_distribuzione_valida()
    test_search_temperature_zero_argmax()
    test_search_env_pulito()
    
    print("\nvisite_to_policy_full():")
    test_policy_full_somma_uno()
    test_policy_full_zero_su_illegali()
    
    print("\ngioca_partita_selfplay():")
    test_selfplay_produce_samples()
    test_selfplay_samples_struttura()
    test_selfplay_simmetria_value()
    
    print("=" * 55)
    print("TUTTI I 9 TEST PASSATI ✓")
