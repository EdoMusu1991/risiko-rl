"""
test_alphazero_backup.py — Test backup MCTS con cambio segno.

Sub-step 3 della Settimana 4. Test mirati a catchare:
- Segno sbagliato (bug piu' comune)
- Path con sequenze NON alternate (Risiko: BLU-BLU-BLU-ROSSO)
- N e W aggiornati correttamente
- Edge case: path con un solo nodo, value zero, value negativo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero.selfplay import Node, backup


def make_node(player: str) -> Node:
    """Helper per creare un nodo pulito."""
    return Node(P=0.0, player_to_move=player)


# ─────────────────────────────────────────────────────────────────
#  TEST CASI BASE
# ─────────────────────────────────────────────────────────────────

def test_backup_path_singolo():
    """Path di un solo nodo: foglia stessa = root."""
    n = make_node("BLU")
    backup([n], value=0.5, player_leaf="BLU")
    assert n.N == 1
    assert n.W == 0.5
    print("  ✓ Path singolo nodo, stesso giocatore foglia")


def test_backup_path_singolo_avversario():
    """
    Edge case strano ma valido: path singolo dove il "player_leaf" e'
    diverso dal nodo. Significa che hai impostato male player_leaf,
    ma il backup deve ancora gestirlo coerentemente.
    """
    n = make_node("BLU")
    backup([n], value=0.5, player_leaf="ROSSO")
    # Il nodo BLU vede valore "negativo" rispetto a ROSSO
    assert n.N == 1
    assert n.W == -0.5
    print("  ✓ Path singolo, player_leaf diverso → segno invertito")


def test_backup_path_due_giocatori_alternati():
    """Path classico stile scacchi: BLU → ROSSO."""
    n0 = make_node("BLU")    # root, BLU muove
    n1 = make_node("ROSSO")  # dopo mossa di BLU, tocca a ROSSO
    
    # Foglia e' n1 (ROSSO muove). value=+0.8 dal POV di ROSSO
    backup([n0, n1], value=0.8, player_leaf="ROSSO")
    
    # n1 e' della foglia (ROSSO) → +value
    assert n1.N == 1
    assert n1.W == 0.8
    
    # n0 (BLU) → -value (quello buono per ROSSO e' cattivo per BLU)
    assert n0.N == 1
    assert n0.W == -0.8
    print("  ✓ Path BLU→ROSSO con value positivo")


# ─────────────────────────────────────────────────────────────────
#  TEST RISIKO-SPECIFICO (sequenze non alternate)
# ─────────────────────────────────────────────────────────────────

def test_backup_sequenze_lunghe_stesso_giocatore():
    """
    CRITICO: Risiko ha sequenze tipo BLU-BLU-BLU-BLU-ROSSO.
    Se uso "track last_player e inverti quando cambia", funziona.
    Se invece confronto sempre con player_leaf, funziona meglio.
    Test: path BLU-BLU-BLU-BLU. value=+0.6 dal POV di BLU.
    Tutti i nodi devono avere W=+0.6.
    """
    nodes = [make_node("BLU") for _ in range(4)]
    
    backup(nodes, value=0.6, player_leaf="BLU")
    
    for i, n in enumerate(nodes):
        assert n.N == 1, f"nodo {i}: N={n.N}"
        assert abs(n.W - 0.6) < 1e-9, f"nodo {i}: W={n.W} (atteso +0.6)"
    print("  ✓ Path BLU×4 con value+0.6 → tutti W=+0.6")


def test_backup_risiko_realistico():
    """
    Path realistico Risiko: BLU-BLU-BLU-ROSSO-ROSSO-ROSSO.
    BLU fa 3 sotto-decisioni (rinforzi, attacco, spostamento), poi
    tocca a ROSSO che ne fa 3.
    
    Foglia: ROSSO (player_leaf). value=+0.7 dal POV di ROSSO.
    
    Aspettativa:
    - 3 nodi BLU → W = -0.7 (per BLU e' cattivo)
    - 3 nodi ROSSO → W = +0.7 (per ROSSO e' buono)
    """
    path = [
        make_node("BLU"),    # root
        make_node("BLU"),
        make_node("BLU"),
        make_node("ROSSO"),
        make_node("ROSSO"),
        make_node("ROSSO"),  # foglia
    ]
    
    backup(path, value=0.7, player_leaf="ROSSO")
    
    for i in range(3):  # nodi BLU
        assert abs(path[i].W - (-0.7)) < 1e-9, \
            f"nodo BLU {i}: W={path[i].W} atteso -0.7"
    for i in range(3, 6):  # nodi ROSSO
        assert abs(path[i].W - 0.7) < 1e-9, \
            f"nodo ROSSO {i-3}: W={path[i].W} atteso +0.7"
    
    # Tutti N=1 (singola visita)
    for n in path:
        assert n.N == 1
    print("  ✓ Path BLU×3 → ROSSO×3 (Risiko realistico)")


def test_backup_risiko_alternanze_irregolari():
    """
    Path piu' patologico: BLU-BLU-ROSSO-BLU-ROSSO-ROSSO-BLU.
    Non e' realistico in pratica (un giocatore non riprende prima
    che l'altro finisca), ma testa la robustezza del confronto
    "con player_leaf" vs "con last_player".
    
    Foglia: BLU. value=+0.4.
    Aspettativa: tutti i nodi BLU → +0.4, tutti ROSSO → -0.4.
    """
    sequence = ["BLU", "BLU", "ROSSO", "BLU", "ROSSO", "ROSSO", "BLU"]
    path = [make_node(p) for p in sequence]
    
    backup(path, value=0.4, player_leaf="BLU")
    
    for i, p in enumerate(sequence):
        atteso = 0.4 if p == "BLU" else -0.4
        assert abs(path[i].W - atteso) < 1e-9, \
            f"nodo {i} ({p}): W={path[i].W} atteso {atteso}"
    print("  ✓ Path con alternanze irregolari")


# ─────────────────────────────────────────────────────────────────
#  TEST ACCUMULO (multiple backup sullo stesso albero)
# ─────────────────────────────────────────────────────────────────

def test_backup_accumulo_multiple_visite():
    """
    Backup chiamato piu' volte sullo stesso path (con value diversi)
    deve sommare correttamente W e contare N.
    """
    n_blu   = make_node("BLU")
    n_rosso = make_node("ROSSO")
    path = [n_blu, n_rosso]
    
    # Visita 1: ROSSO foglia, value=+0.8
    backup(path, value=0.8, player_leaf="ROSSO")
    assert n_rosso.N == 1 and abs(n_rosso.W - 0.8) < 1e-9
    assert n_blu.N == 1   and abs(n_blu.W - (-0.8)) < 1e-9
    
    # Visita 2: ROSSO foglia, value=+0.4
    backup(path, value=0.4, player_leaf="ROSSO")
    assert n_rosso.N == 2 and abs(n_rosso.W - 1.2) < 1e-9
    assert n_blu.N == 2   and abs(n_blu.W - (-1.2)) < 1e-9
    
    # Visita 3: ROSSO foglia, value=-0.3 (perdita)
    backup(path, value=-0.3, player_leaf="ROSSO")
    assert n_rosso.N == 3 and abs(n_rosso.W - 0.9) < 1e-9
    # Per BLU: -(-0.3) = +0.3 → W = -1.2 + 0.3 = -0.9
    assert n_blu.N == 3 and abs(n_blu.W - (-0.9)) < 1e-9
    
    # Q dei due nodi
    assert abs(n_rosso.Q - 0.3) < 1e-9
    assert abs(n_blu.Q - (-0.3)) < 1e-9
    print("  ✓ Accumulo multiple visite (3 backup)")


def test_backup_value_zero():
    """Value=0 lascia W invariato ma incrementa N."""
    n = make_node("BLU")
    n.N = 5
    n.W = 2.5
    
    backup([n], value=0.0, player_leaf="BLU")
    assert n.N == 6
    assert n.W == 2.5
    print("  ✓ Value=0 incrementa N ma non cambia W")


def test_backup_value_negativo():
    """Value negativo (la foglia perde) propaga inversamente."""
    n_blu   = make_node("BLU")
    n_rosso = make_node("ROSSO")
    
    # ROSSO foglia, value=-0.5 (ROSSO sta perdendo)
    backup([n_blu, n_rosso], value=-0.5, player_leaf="ROSSO")
    
    assert abs(n_rosso.W - (-0.5)) < 1e-9
    # Per BLU, "ROSSO perde" e' positivo → +0.5
    assert abs(n_blu.W - 0.5) < 1e-9
    print("  ✓ Value negativo: ROSSO perde, BLU guadagna")


# ─────────────────────────────────────────────────────────────────
#  TEST COERENZA Q
# ─────────────────────────────────────────────────────────────────

def test_backup_coerenza_Q():
    """
    Se la foglia ROSSO vede sempre value=+0.6 (vincente per ROSSO),
    dopo molte visite:
    - Q(nodo_rosso) → +0.6
    - Q(nodo_blu)   → -0.6
    """
    n_blu   = make_node("BLU")
    n_rosso = make_node("ROSSO")
    
    for _ in range(100):
        backup([n_blu, n_rosso], value=0.6, player_leaf="ROSSO")
    
    assert abs(n_rosso.Q - 0.6) < 1e-9, f"Q ROSSO = {n_rosso.Q}"
    assert abs(n_blu.Q - (-0.6)) < 1e-9, f"Q BLU = {n_blu.Q}"
    print("  ✓ Q converge correttamente per entrambi i giocatori")


# ─────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
#  TEST "PERFETTI" DI CHATGPT (validati Settimana 4)
#  Becca i bug semantici nascosti: significato del value lungo il path
# ─────────────────────────────────────────────────────────────────

def test_chatgpt_1_semantica_valore():
    """
    CRITICO: il valore viene interpretato correttamente dal POV di ogni nodo,
    anche con sequenza Risiko-style (BLU×3 → ROSSO×2).
    """
    path = [
        Node(player_to_move="BLU"),
        Node(player_to_move="BLU"),
        Node(player_to_move="BLU"),
        Node(player_to_move="ROSSO"),
        Node(player_to_move="ROSSO"),
    ]
    backup(path, value=+1.0, player_leaf="ROSSO")
    # ROSSO vede positivo
    assert path[3].W == 1.0
    assert path[4].W == 1.0
    # BLU vede negativo (cattivo per BLU)
    assert path[0].W == -1.0
    assert path[1].W == -1.0
    assert path[2].W == -1.0
    for n in path:
        assert n.N == 1
    print("  ✓ ChatGPT Test 1: semantica valore (BLU×3 → ROSSO×2)")


def test_chatgpt_2_value_negativo():
    """ROSSO perde (value=-1) → BLU vede +1."""
    path = [Node(player_to_move="BLU"), Node(player_to_move="ROSSO")]
    backup(path, value=-1.0, player_leaf="ROSSO")
    assert path[1].W == -1.0
    assert path[0].W == +1.0
    print("  ✓ ChatGPT Test 2: value negativo")


def test_chatgpt_3_accumulo_coerente():
    """Doppio backup: W e N accumulano sui due giocatori coerentemente."""
    path = [Node(player_to_move="BLU"), Node(player_to_move="ROSSO")]
    backup(path, +1.0, "ROSSO")
    backup(path, +1.0, "ROSSO")
    assert path[1].W == 2.0 and path[1].N == 2
    assert path[0].W == -2.0 and path[0].N == 2
    print("  ✓ ChatGPT Test 3: doppio backup coerente")


def test_chatgpt_4_ordine_path():
    """Sequenza BLU-ROSSO-BLU(foglia): root BLU vede +1, mid ROSSO vede -1."""
    path = [
        Node(player_to_move="BLU"),
        Node(player_to_move="ROSSO"),
        Node(player_to_move="BLU"),
    ]
    backup(path, +1.0, "BLU")
    assert path[2].W == 1.0
    assert path[1].W == -1.0
    assert path[0].W == 1.0
    print("  ✓ ChatGPT Test 4: ordine path BLU-ROSSO-BLU(foglia)")


# ─────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test backup MCTS (alphazero/selfplay/backup.py)")
    print("=" * 55)
    
    print("\nCasi base:")
    test_backup_path_singolo()
    test_backup_path_singolo_avversario()
    test_backup_path_due_giocatori_alternati()
    
    print("\nRisiko-specifico (sequenze non alternate):")
    test_backup_sequenze_lunghe_stesso_giocatore()
    test_backup_risiko_realistico()
    test_backup_risiko_alternanze_irregolari()
    
    print("\nAccumulo e edge cases:")
    test_backup_accumulo_multiple_visite()
    test_backup_value_zero()
    test_backup_value_negativo()
    
    print("\nCoerenza:")
    test_backup_coerenza_Q()
    
    print("\nTest 'perfetti' di ChatGPT (semantica valore):")
    test_chatgpt_1_semantica_valore()
    test_chatgpt_2_value_negativo()
    test_chatgpt_3_accumulo_coerente()
    test_chatgpt_4_ordine_path()
    
    print("=" * 55)
    print("TUTTI I 14 TEST PASSATI ✓")
