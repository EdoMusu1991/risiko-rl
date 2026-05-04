"""
test_alphazero_simulate.py — Test simulate() MCTS+rete.

I 6 test "perfetti" forniti da ChatGPT, adattati alle API del nostro env:
- env.giocatore_corrente_colore() → env.stato.giocatore_corrente
- env.action_masks() → info["action_mask"] da reset/step

Sub-step 4 della Settimana 4. Validazione finale prima di passare al loop
self-play (sub-step 5).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from risiko_env import encoding as e
e.STAGE_A_ATTIVO = True
from risiko_env import RisikoEnv

from alphazero.network import RisikoNet
from alphazero.selfplay import Node, simulate


def make_root_and_env(seed=42):
    """Helper: crea env, reset, e root MCTS pronta per simulate."""
    env = RisikoEnv(seed=seed, mode_1v1=True, reward_mode='margin')
    obs, info = env.reset()
    root = Node(
        snapshot=env.snapshot(),
        player_to_move=env.stato.giocatore_corrente,
        P=1.0,
    )
    return env, root


# ─────────────────────────────────────────────────────────────────
#  TEST 1 (CRITICO): simulate non sporca l'env
# ─────────────────────────────────────────────────────────────────

def test_chatgpt_1_simulate_non_modifica_env():
    """
    Se questo fallisce, simulate() lascia l'env in uno stato sbagliato.
    E' il bug PIU' GRAVE possibile: rompe le simulazioni successive.
    """
    env, root = make_root_and_env(seed=42)
    snap_before = env.snapshot()
    
    net = RisikoNet()
    simulate(root, env, net)
    
    snap_after = env.snapshot()
    assert snap_after == snap_before, "simulate() ha sporcato l'env!"
    print("  ✓ Test 1: simulate non modifica env (CRITICO)")


# ─────────────────────────────────────────────────────────────────
#  TEST 2: root viene visitata ed espansa
# ─────────────────────────────────────────────────────────────────

def test_chatgpt_2_simulate_espande_root():
    env, root = make_root_and_env(seed=42)
    
    # Stato iniziale: nodo non visitato, nessun figlio
    assert root.N == 0
    assert len(root.children) == 0
    assert root.legal_actions is None
    
    net = RisikoNet()
    simulate(root, env, net)
    
    assert root.N == 1, f"root.N={root.N}, atteso 1"
    assert len(root.children) > 0, f"root non ha figli!"
    assert root.legal_actions is not None
    print(f"  ✓ Test 2: root espansa con {len(root.children)} figli")


# ─────────────────────────────────────────────────────────────────
#  TEST 3: tutti i figli sono azioni legali
# ─────────────────────────────────────────────────────────────────

def test_chatgpt_3_simulate_crea_solo_figli_legali():
    env, root = make_root_and_env(seed=42)
    
    # Salvo la mask prima di simulate (perche' simulate ripristina lo stato)
    info = env._costruisci_info()
    mask = info["action_mask"]
    legal_actions = set(np.where(mask)[0])
    
    net = RisikoNet()
    simulate(root, env, net)
    
    # Tutti i figli devono essere azioni legali
    children_actions = set(root.children.keys())
    assert children_actions.issubset(legal_actions), \
        f"Figli non legali: {children_actions - legal_actions}"
    print(f"  ✓ Test 3: tutti i {len(children_actions)} figli sono azioni legali")


# ─────────────────────────────────────────────────────────────────
#  TEST 4: prior validi, sommano a 1
# ─────────────────────────────────────────────────────────────────

def test_chatgpt_4_simulate_prior_figli_sommano_a_uno():
    env, root = make_root_and_env(seed=42)
    
    net = RisikoNet()
    simulate(root, env, net)
    
    total_prior = sum(child.P for child in root.children.values())
    assert abs(total_prior - 1.0) < 1e-5, f"Sum priors = {total_prior}"
    assert all(child.P >= 0 for child in root.children.values()), \
        "Prior negativo trovato!"
    print(f"  ✓ Test 4: prior figli sommano a {total_prior:.6f} (~1.0)")


# ─────────────────────────────────────────────────────────────────
#  TEST 5: child snapshot valido (ripristinabile)
# ─────────────────────────────────────────────────────────────────

def test_chatgpt_5_simulate_child_snapshot_ripristinabile():
    env, root = make_root_and_env(seed=42)
    
    net = RisikoNet()
    simulate(root, env, net)
    
    first_child = next(iter(root.children.values()))
    assert first_child.snapshot is not None, "child senza snapshot!"
    
    env.restore(first_child.snapshot)
    snap_post = env.snapshot()
    assert snap_post == first_child.snapshot, \
        "Snapshot del figlio non ripristinabile coerentemente!"
    print("  ✓ Test 5: child snapshot ripristinabile")


# ─────────────────────────────────────────────────────────────────
#  TEST 6: seconda simulate scende nell'albero
# ─────────────────────────────────────────────────────────────────

def test_chatgpt_6_seconda_simulate_visita_figlio():
    env, root = make_root_and_env(seed=42)
    
    net = RisikoNet()
    simulate(root, env, net)
    simulate(root, env, net)
    
    assert root.N == 2, f"root.N={root.N}, atteso 2"
    visited_children = [c for c in root.children.values() if c.N > 0]
    assert len(visited_children) >= 1, \
        f"Nessun figlio visitato dopo 2 simulazioni!"
    print(f"  ✓ Test 6: seconda simulate scende ({len(visited_children)} figli visitati)")


# ─────────────────────────────────────────────────────────────────
#  TEST EXTRA: stress test con piu' simulazioni
# ─────────────────────────────────────────────────────────────────

def test_extra_50_simulazioni_stato_consistente():
    """
    Test extra: 50 simulazioni di seguito.
    Verifica che N cresca linearmente e env resti pulito.
    """
    env, root = make_root_and_env(seed=42)
    snap_iniziale = env.snapshot()
    
    net = RisikoNet()
    for _ in range(50):
        simulate(root, env, net)
    
    assert root.N == 50, f"root.N={root.N}, atteso 50"
    assert env.snapshot() == snap_iniziale, "env sporcato dopo 50 simulazioni!"
    
    # Distribuzione di visite sui figli (informativa)
    visits = sorted(
        [c.N for c in root.children.values()],
        reverse=True,
    )
    print(f"  ✓ Extra: 50 sim, N=50, env pulito. Top 5 visite: {visits[:5]}")


def test_extra_value_propagato_a_root():
    """
    Dopo molte simulazioni, root.Q dovrebbe essere un numero
    finito in [-1, +1]. Test sanity.
    """
    env, root = make_root_and_env(seed=42)
    net = RisikoNet()
    for _ in range(20):
        simulate(root, env, net)
    
    Q = root.Q
    assert np.isfinite(Q), f"root.Q non finito: {Q}"
    assert -1.0 <= Q <= 1.0, f"root.Q fuori [-1,1]: {Q}"
    print(f"  ✓ Extra: dopo 20 sim, root.Q = {Q:+.4f} (finito, in range)")


def test_extra_stato_con_molte_azioni():
    """
    Test in uno stato con MOLTE azioni legali (fase rinforzo).
    Atteso: 21+ figli, prior distribuita, MCTS visita figli diversi.
    """
    env = RisikoEnv(seed=42, mode_1v1=True, reward_mode='margin')
    obs, info = env.reset()
    
    # Avanza fino a fase rinforzo (il primo step con tante azioni)
    for _ in range(5):  # max 5 step per arrivare a una fase con piu' azioni
        legali = np.where(info["action_mask"])[0]
        if env.sotto_fase == "rinforzo" and len(legali) > 5:
            break
        if len(legali) == 0:
            break
        # Scegli prima azione legale per avanzare
        obs, _, term, trunc, info = env.step(int(legali[0]))
        if term or trunc:
            break
    
    print(f"     (sotto_fase={env.sotto_fase}, n_legali={info['action_mask'].sum()})")
    
    root = Node(
        snapshot=env.snapshot(),
        player_to_move=env.stato.giocatore_corrente,
        P=1.0,
    )
    snap_iniziale = env.snapshot()
    
    net = RisikoNet()
    for _ in range(100):
        simulate(root, env, net)
    
    assert root.N == 100
    assert env.snapshot() == snap_iniziale, "env sporcato!"
    
    # In stato con tante azioni legali, dopo 100 sim aspetto:
    # - molti figli (>5)
    # - distribuzione di visite non degenere (non tutta su 1)
    n_children = len(root.children)
    visits = sorted([c.N for c in root.children.values()], reverse=True)
    n_visited = sum(1 for c in root.children.values() if c.N > 0)
    
    print(f"  ✓ Extra: stato 'reale' ({n_children} figli)")
    print(f"     Visite top 3: {visits[:3]}")
    print(f"     Figli visitati: {n_visited} / {n_children}")
    
    # Asserzioni morbide perche' la rete random e' rumore
    assert n_children >= 5, f"Troppi pochi figli: {n_children}"
    # Almeno 5 figli devono essere stati visitati (esplorazione MCTS)
    assert n_visited >= 5, f"Solo {n_visited} figli visitati su {n_children}"


# ─────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test simulate() MCTS+rete (alphazero/selfplay/simulate.py)")
    print("=" * 60)
    
    print("\nI 6 test 'perfetti' di ChatGPT:")
    test_chatgpt_1_simulate_non_modifica_env()
    test_chatgpt_2_simulate_espande_root()
    test_chatgpt_3_simulate_crea_solo_figli_legali()
    test_chatgpt_4_simulate_prior_figli_sommano_a_uno()
    test_chatgpt_5_simulate_child_snapshot_ripristinabile()
    test_chatgpt_6_seconda_simulate_visita_figlio()
    
    print("\nTest extra (stress + sanity):")
    test_extra_50_simulazioni_stato_consistente()
    test_extra_value_propagato_a_root()
    test_extra_stato_con_molte_azioni()
    
    print("=" * 60)
    print("TUTTI I 9 TEST PASSATI ✓")
