"""
test_alphazero_node.py — Test per Node MCTS e PUCT selection.

Sub-step 1+2 della Settimana 4 (validati da ChatGPT).
NON include simulate o backup (sub-step 3-4).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np

from alphazero.selfplay import Node, select_child, select_action_from_root


# ─────────────────────────────────────────────────────────────────
#  TEST NODE CLASS
# ─────────────────────────────────────────────────────────────────

def test_node_creazione():
    """Node si crea con default ragionevoli."""
    n = Node()
    assert n.N == 0
    assert n.W == 0.0
    assert n.P == 0.0
    assert n.children == {}
    assert n.parent is None
    assert n.action_taken is None
    assert n.snapshot is None
    assert n.legal_actions is None
    assert not n.is_terminal
    assert n.terminal_value == 0.0
    assert n.player_to_move is None
    print("  ✓ Node default creation")


def test_node_con_attributi():
    """Node si crea con attributi specifici."""
    parent = Node(player_to_move="BLU")
    child = Node(P=0.3, parent=parent, action_taken=42, player_to_move="ROSSO")
    assert child.P == 0.3
    assert child.parent is parent
    assert child.action_taken == 42
    assert child.player_to_move == "ROSSO"
    print("  ✓ Node creation con attributi")


def test_node_Q():
    """Q = W/N se N>0, 0 altrimenti."""
    n = Node()
    assert n.Q == 0.0  # mai visitato
    n.N = 4
    n.W = 2.0
    assert n.Q == 0.5
    n.W = -1.0
    assert n.Q == -0.25
    print("  ✓ Node.Q calcolato correttamente")


def test_node_is_expanded():
    """is_expanded() vero solo se ha children."""
    n = Node()
    assert not n.is_expanded()
    assert n.is_leaf()
    n.children[0] = Node()
    assert n.is_expanded()
    assert not n.is_leaf()
    print("  ✓ Node.is_expanded / is_leaf")


def test_node_repr():
    """__repr__ produce stringa leggibile."""
    n = Node(P=0.25, action_taken=7, player_to_move="BLU")
    s = repr(n)
    assert "BLU" in s
    assert "P=0.250" in s
    assert "action_taken=7" in s
    print("  ✓ Node.__repr__")


def test_node_slots():
    """__slots__ funziona — non si possono aggiungere attributi extra."""
    n = Node()
    try:
        n.attributo_inesistente = 42
        assert False, "Avrebbe dovuto sollevare AttributeError"
    except AttributeError:
        pass
    print("  ✓ Node usa __slots__ (no attributi spuri)")


# ─────────────────────────────────────────────────────────────────
#  TEST SELECT_CHILD (PUCT)
# ─────────────────────────────────────────────────────────────────

def test_select_child_no_children():
    """select_child su nodo senza figli ritorna None."""
    n = Node()
    assert select_child(n) is None
    print("  ✓ select_child su nodo senza figli → None")


def test_select_child_un_figlio():
    """select_child con un solo figlio lo restituisce."""
    parent = Node(player_to_move="BLU")
    parent.N = 1
    child = Node(P=0.5, parent=parent, action_taken=0, player_to_move="ROSSO")
    parent.children[0] = child
    
    result = select_child(parent)
    assert result is child
    print("  ✓ select_child con 1 figlio")


def test_select_child_prior_decide_se_mai_visitati():
    """
    Quando tutti i figli hanno N=0, vince quello con prior maggiore.
    Q=0 per tutti, U = c * P * sqrt(parent.N) / 1
    Quindi vince P max.
    """
    parent = Node(player_to_move="BLU")
    parent.N = 4  # padre visitato 4 volte
    
    # 3 figli con prior diversi, mai visitati
    child_alto_P  = Node(P=0.7, parent=parent, action_taken=0, player_to_move="ROSSO")
    child_medio_P = Node(P=0.2, parent=parent, action_taken=1, player_to_move="ROSSO")
    child_basso_P = Node(P=0.1, parent=parent, action_taken=2, player_to_move="ROSSO")
    
    parent.children[0] = child_alto_P
    parent.children[1] = child_medio_P
    parent.children[2] = child_basso_P
    
    # Tutti N=0, dovrebbe vincere alto P
    chosen = select_child(parent, c_puct=1.5)
    assert chosen is child_alto_P, f"Atteso child_alto_P, ottenuto {chosen}"
    print("  ✓ select_child: con prior diversi e N=0, vince P max")


def test_select_child_q_decide_se_visitati():
    """
    Quando un figlio e' molto visitato e ha Q alto, U decresce e
    a un certo punto un altro figlio con prior alto puo' vincere.
    Test base: figlio con Q alto e poche visite vince su Q basso piu' visitato.
    """
    parent = Node(player_to_move="BLU")
    parent.N = 100
    
    # Figlio "vincente": Q alto
    c1 = Node(P=0.3, parent=parent, action_taken=0, player_to_move="ROSSO")
    c1.N = 10
    c1.W = 8.0   # Q = 0.8
    
    # Figlio "perdente": Q basso, anche se prior simile
    c2 = Node(P=0.3, parent=parent, action_taken=1, player_to_move="ROSSO")
    c2.N = 10
    c2.W = -5.0  # Q = -0.5
    
    parent.children[0] = c1
    parent.children[1] = c2
    
    chosen = select_child(parent, c_puct=1.5)
    assert chosen is c1
    print("  ✓ select_child: tra figli simili, vince Q max")


def test_select_child_PUCT_formula():
    """Verifica esplicita della formula PUCT."""
    parent = Node(player_to_move="BLU")
    parent.N = 9  # sqrt(9) = 3
    
    c_a = Node(P=0.5, parent=parent, action_taken=0, player_to_move="ROSSO")
    c_a.N = 2
    c_a.W = 1.0  # Q = 0.5
    
    c_b = Node(P=0.5, parent=parent, action_taken=1, player_to_move="ROSSO")
    c_b.N = 0
    
    parent.children[0] = c_a
    parent.children[1] = c_b
    
    c_puct = 2.0
    
    # Score atteso A: Q=0.5 + 2.0 * 0.5 * 3/(1+2) = 0.5 + 1.0 = 1.5
    # Score atteso B: Q=0.0 + 2.0 * 0.5 * 3/(1+0) = 0.0 + 3.0 = 3.0
    chosen = select_child(parent, c_puct=c_puct)
    assert chosen is c_b, "PUCT favorisce esplorazione di nodi mai visitati"
    print("  ✓ select_child: formula PUCT calcolata correttamente")


# ─────────────────────────────────────────────────────────────────
#  TEST SELECT_ACTION_FROM_ROOT
# ─────────────────────────────────────────────────────────────────

def test_select_action_temperature_zero():
    """Temperature=0 => argmax deterministico delle visite."""
    root = Node(player_to_move="BLU")
    root.N = 100
    
    c0 = Node(P=0.3, parent=root, action_taken=0, player_to_move="ROSSO"); c0.N = 5
    c1 = Node(P=0.3, parent=root, action_taken=5, player_to_move="ROSSO"); c1.N = 80
    c2 = Node(P=0.3, parent=root, action_taken=10, player_to_move="ROSSO"); c2.N = 15
    root.children = {0: c0, 5: c1, 10: c2}
    
    rng = np.random.default_rng(42)
    action, dist = select_action_from_root(root, temperature=0, rng=rng)
    assert action == 5, f"Atteso 5 (max visite), ottenuto {action}"
    
    # Distribuzione one-hot
    actions, probs = zip(*dist)
    assert sum(probs) == 1.0
    assert max(probs) == 1.0
    print("  ✓ select_action T=0: argmax")


def test_select_action_temperature_uno():
    """Temperature=1 => sampling proporzionale alle visite."""
    root = Node(player_to_move="BLU")
    root.N = 100
    
    c0 = Node(P=0.3, parent=root, action_taken=0, player_to_move="ROSSO"); c0.N = 10
    c1 = Node(P=0.3, parent=root, action_taken=1, player_to_move="ROSSO"); c1.N = 80
    c2 = Node(P=0.3, parent=root, action_taken=2, player_to_move="ROSSO"); c2.N = 10
    root.children = {0: c0, 1: c1, 2: c2}
    
    # 1000 sample, l'azione 1 dovrebbe vincere ~80% delle volte
    rng = np.random.default_rng(42)
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(1000):
        action, _ = select_action_from_root(root, temperature=1.0, rng=rng)
        counts[action] += 1
    
    # Aspetto: ~10% / 80% / 10%
    assert 50 < counts[0] < 150, f"Counts azione 0: {counts[0]}"
    assert 700 < counts[1] < 900, f"Counts azione 1: {counts[1]}"
    assert 50 < counts[2] < 150, f"Counts azione 2: {counts[2]}"
    print(f"  ✓ select_action T=1: sampling ~ visite (counts={counts})")


def test_select_action_dist_e_visite():
    """La distribuzione restituita riflette le visite."""
    root = Node(player_to_move="BLU")
    root.N = 100
    
    c0 = Node(P=0.3, action_taken=0); c0.N = 25
    c1 = Node(P=0.3, action_taken=1); c1.N = 50
    c2 = Node(P=0.3, action_taken=2); c2.N = 25
    root.children = {0: c0, 1: c1, 2: c2}
    
    rng = np.random.default_rng(0)
    _, dist = select_action_from_root(root, temperature=1.0, rng=rng)
    
    # Distribuzione con T=1: proporzionale alle visite
    actions, probs = zip(*dist)
    assert abs(probs[0] - 0.25) < 1e-9, f"prob[0]={probs[0]}"
    assert abs(probs[1] - 0.50) < 1e-9
    assert abs(probs[2] - 0.25) < 1e-9
    assert abs(sum(probs) - 1.0) < 1e-9
    print("  ✓ select_action: distribuzione = visite normalizzate (T=1)")


# ─────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test Node + Selection (alphazero/selfplay/)")
    print("=" * 55)
    
    print("\nNode class:")
    test_node_creazione()
    test_node_con_attributi()
    test_node_Q()
    test_node_is_expanded()
    test_node_repr()
    test_node_slots()
    
    print("\nselect_child (PUCT):")
    test_select_child_no_children()
    test_select_child_un_figlio()
    test_select_child_prior_decide_se_mai_visitati()
    test_select_child_q_decide_se_visitati()
    test_select_child_PUCT_formula()
    
    print("\nselect_action_from_root:")
    test_select_action_temperature_zero()
    test_select_action_temperature_uno()
    test_select_action_dist_e_visite()
    
    print("=" * 55)
    print("TUTTI I 14 TEST PASSATI ✓")
