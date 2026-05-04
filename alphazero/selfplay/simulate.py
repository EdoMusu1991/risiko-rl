"""
simulate.py — Singola simulazione MCTS guidata dalla rete neurale.

Sub-step 4 della Settimana 4 (validato da ChatGPT).

Una simulazione fa:
1. SELECTION  — scendi nell'albero seguendo PUCT finche' trovi una foglia
2. EXPANSION  — se la foglia non e' terminale, crea TUTTI i figli legali
                 (con prior dato dalla rete)
3. EVALUATION — il value della foglia viene dalla rete (in fase 2 sara'
                 mix con rollout euristico per warm-start)
4. BACKUP     — propaga il value lungo il path con cambio segno per nodi
                 di giocatori diversi

REGOLA D'ORO (ChatGPT):
- Alla fine di simulate(), env DEVE essere ripristinato al snapshot della root
- Anche in caso di errore (try/finally)
- Senza questo, simulazioni successive sono corrotte

NB: questa e' la versione "pura" che usa SOLO la rete per il value.
Il warm-start con rollout euristico verra' aggiunto in sub-step 5+.
"""

from __future__ import annotations
from typing import Optional, Callable
import numpy as np
import torch

from .node import Node
from .selection import select_child
from .backup import backup
from ..network import RisikoNet, apply_mask_and_softmax


def simulate(
    root: Node,
    env,
    net: RisikoNet,
    c_puct: float = 1.5,
    rollout_value_fn: Optional[Callable] = None,
) -> None:
    """
    Esegue UNA simulazione MCTS dalla root.
    
    Args:
        root: Node radice. DEVE avere:
            - root.snapshot impostato (stato iniziale)
            - root.player_to_move impostato
        env: RisikoEnv (verra' restituito allo stato della root al termine)
        net: RisikoNet per policy/value
        c_puct: coefficiente di esplorazione PUCT (1.5 default)
        rollout_value_fn: opzionale, se fornito viene usata al posto del
            value della rete per la EVALUATION. Per warm-start.
    
    Modifica root e suoi discendenti in-place. Non ritorna nulla.
    """
    try:
        # ════════════════════════════════════════════════════════════
        # 1. SELECTION — scendi finche' trovi una foglia
        # ════════════════════════════════════════════════════════════
        env.restore(root.snapshot)
        node = root
        path = [node]
        
        # Scendi finche' il nodo ha figli (e' espanso) e non e' terminale
        while node.is_expanded() and not node.is_terminal:
            child = select_child(node, c_puct=c_puct)
            if child is None:
                break  # safety: non dovrebbe succedere se is_expanded()
            
            # Esegui l'azione che porta al child
            env.step(child.action_taken)
            
            node = child
            path.append(node)
        
        # ════════════════════════════════════════════════════════════
        # 2. EXPANSION + EVALUATION
        # ════════════════════════════════════════════════════════════
        if node.is_terminal:
            # Foglia terminale: usa il valore terminale gia' calcolato
            value = node.terminal_value
            leaf_player = node.player_to_move
        else:
            # Foglia non terminale: chiama la rete UNA VOLTA
            obs = env._costruisci_observation()  # observation in [0,1]
            info = env._costruisci_info()
            mask_np = info["action_mask"]
            
            # Sposta i tensori sullo stesso device della rete
            net_device = next(net.parameters()).device
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(net_device)
            mask_t = torch.from_numpy(mask_np).bool().to(net_device)
            
            with torch.no_grad():
                policy_logits, value_tensor = net(obs_t)
                policy_dist = apply_mask_and_softmax(policy_logits[0], mask_t)
            
            value = float(value_tensor.item())  # scalar
            
            # Sposta policy_dist su CPU per accesso indicizzato veloce
            policy_dist_cpu = policy_dist.cpu()
            
            # Espandi: crea TUTTI i figli per le azioni legali
            legal_actions = np.where(mask_np)[0].tolist()
            node.legal_actions = legal_actions
            
            # Salva snapshot del nodo PRIMA di toccare l'env per i figli
            # (potrebbe gia' essere salvato per la root, ma per nodi
            # interni va impostato qui)
            if node.snapshot is None:
                node.snapshot = env.snapshot()
            snap_node = node.snapshot
            
            for action in legal_actions:
                # Esegui l'azione per ottenere lo stato del figlio
                env.restore(snap_node)
                obs_new, reward, term, trunc, info_new = env.step(int(action))
                child_player = env.stato.giocatore_corrente
                
                child = Node(
                    P=float(policy_dist_cpu[action].item()),
                    parent=node,
                    action_taken=int(action),
                    player_to_move=child_player,
                    snapshot=env.snapshot(),
                )
                
                if term or trunc:
                    child.is_terminal = True
                    child.terminal_value = float(reward)
                
                node.children[int(action)] = child
            
            # Ripristina lo snapshot della foglia per coerenza
            env.restore(snap_node)
            
            leaf_player = node.player_to_move
            
            # Se rollout_value_fn fornita, sovrascrive il value della rete
            # (warm-start). Da implementare in sub-step 5.
            if rollout_value_fn is not None:
                value = rollout_value_fn(env)
        
        # ════════════════════════════════════════════════════════════
        # 3. BACKUP — propaga il value lungo il path
        # ════════════════════════════════════════════════════════════
        backup(path, value, leaf_player)
    
    finally:
        # ════════════════════════════════════════════════════════════
        # REGOLA D'ORO (ChatGPT): env SEMPRE allo stato della root al termine
        # ════════════════════════════════════════════════════════════
        env.restore(root.snapshot)


# ═════════════════════════════════════════════════════════════════════════
#  PR2 — SIMULATE SIMMETRICO (fix MCTS: turni avversari come livelli MIN)
# ═════════════════════════════════════════════════════════════════════════
#
# Il problema risolto
# -------------------
# La funzione simulate() originale usa un solo env. Quando dentro un rollout
# MCTS il turno passa al giocatore avversario, env.step() esplode con
# l'assert "step() chiamato fuori turno bot. Corrente: ROSSO".
#
# Anche risolto l'assert (es. con _skip_giro_avversari, fatto da PR1), resta
# un bug di SEGNO latente: l'observation costruita da env.bot_color e' BLU-
# centric anche quando MCTS espande nodi del giocatore ROSSO, quindi la value
# head riceve un input dal POV sbagliato.
#
# Il design
# ---------
# Estendiamo il pattern di PR1 (due env templati) DENTRO simulate. Ogni nodo
# MCTS porta con se' un player_to_move e uno snapshot "compatibile" con
# l'env di quel colore (cioe' con sotto_fase=tris armata, _combinazioni_tris
# del colore giusto, ecc.).
#
# INVARIANTE CRITICA:
#   Per ogni Node n, deve valere:
#     n.snapshot e' stato preso da envs[n.player_to_move]
#     dopo aver eseguito _inizia_fase_tris() su quell'env
#   (Eccezione: nodi terminali, che hanno snapshot=None e terminal_value
#   settato.)
#
# Conseguenza: al restore di un nodo, restoriamo su envs[n.player_to_move],
# riallineiamo l'altro env (alias stato/rng), e siamo "pronti a giocare".
#
# Cambio turno
# ------------
# Quando env.step() fa terminare il turno (sotto_fase=None) e la partita non
# e' finita, dobbiamo:
#   1. switch env_attivo all'env dell'altro colore
#   2. riallineare stato/rng (alias)
#   3. chiamare _inizia_fase_tris() sul nuovo env_attivo
# Il child viene snapshottato DOPO questi tre passi (Opzione Y, scelta
# concordata): cosi' lo snapshot del child e' compatibile con
# envs[child.player_to_move] e l'invariante e' rispettata.
#
# Cosa NON facciamo
# -----------------
# - Non tocchiamo env.py, encoding.py, backup.py, selection.py, node.py.
# - Non implementiamo 4-player (envs e' un dict 1v1).
# - Non sostituiamo simulate() originale: aggiungiamo simulate_simmetrico
#   accanto. Cosi' i test esistenti (test_alphazero_simulate.py ecc.) non
#   si rompono.

def _restore_e_riallinea(envs: dict, color_attivo: str, snapshot: dict):
    """
    Restora `snapshot` sull'env del colore attivo e riallinea l'altro env
    facendo aliasing di stato/rng. Ritorna l'env attivo (gia' restored).

    Pre-condizione: lo snapshot deve essere stato preso da envs[color_attivo]
    (cioe' compatibile con il bot_color di quell'env). Garantito dall'invariante
    sui Node.
    """
    env_attivo = envs[color_attivo]
    env_attivo.restore(snapshot)
    altro = "ROSSO" if color_attivo == "BLU" else "BLU"
    env_passivo = envs[altro]
    env_passivo.stato = env_attivo.stato
    env_passivo.rng = env_attivo.rng
    return env_attivo


def _switch_se_turno_finito(envs: dict, env_attivo):
    """
    Se env_attivo.sotto_fase e' None (turno appena chiuso) e la partita non e'
    terminata, switcha all'env dell'altro colore: alias stato/rng + arma fase
    tris. Ritorna il nuovo env attivo (potrebbe essere lo stesso se non si
    deve switchare).

    Casi:
    - sotto_fase != None  → niente switch, ritorna env_attivo.
    - sotto_fase == None ma stato.terminata → niente switch, ritorna env_attivo
      (la partita e' finita, non c'e' un "prossimo giocatore"). Il chiamante
      gestira' il terminale.
    - sotto_fase == None e partita continua → switch effettivo.
    """
    if env_attivo.sotto_fase is not None:
        return env_attivo
    if env_attivo.stato.terminata:
        return env_attivo
    altro_color = "ROSSO" if env_attivo.bot_color == "BLU" else "BLU"
    env_passivo = envs[altro_color]
    env_passivo.stato = env_attivo.stato
    env_passivo.rng = env_attivo.rng
    env_passivo._inizia_fase_tris()
    return env_passivo


def simulate_simmetrico(
    root: Node,
    envs: dict,
    net: RisikoNet,
    c_puct: float = 1.5,
    rollout_value_fn: Optional[Callable] = None,
) -> None:
    """
    Versione simmetrica di simulate() per AlphaZero puro (PR2).

    A differenza di simulate() che usa un solo env "BLU-centric", qui usiamo
    DUE env (uno per BLU e uno per ROSSO) e switchiamo l'env attivo ad ogni
    cambio turno. L'observation passata alla rete e' sempre orientata sul
    player_to_move del nodo foglia (perche' env_attivo.bot_color == leaf
    player_to_move per costruzione).

    Args:
        root: Node radice. DEVE avere:
            - root.snapshot impostato e compatibile con envs[root.player_to_move]
            - root.player_to_move in {"BLU", "ROSSO"}
        envs: dict con envs["BLU"] e envs["ROSSO"]. Entrambi devono avere
            _skip_giro_avversari=True (cosi' step non gira automaticamente
            i turni dell'altro). Entrambi vengono ripristinati allo stato
            della root nel finally.
        net: RisikoNet per policy/value.
        c_puct: coefficiente PUCT.
        rollout_value_fn: opzionale, sovrascrive il value della rete (warm-start).

    Modifica root e suoi discendenti in-place. Non ritorna nulla.
    """
    assert root.player_to_move in envs, (
        f"root.player_to_move={root.player_to_move} non in envs={list(envs.keys())}"
    )
    assert "BLU" in envs and "ROSSO" in envs, (
        f"envs deve contenere BLU e ROSSO, trovato: {list(envs.keys())}"
    )

    try:
        # ════════════════════════════════════════════════════════════
        # 1. SELECTION — scendi finche' trovi una foglia
        # ════════════════════════════════════════════════════════════
        env_attivo = _restore_e_riallinea(envs, root.player_to_move, root.snapshot)
        node = root
        path = [node]

        while node.is_expanded() and not node.is_terminal:
            child = select_child(node, c_puct=c_puct)
            if child is None:
                break  # safety: non dovrebbe succedere se is_expanded()

            # Pre-step: env_attivo deve corrispondere al player_to_move del nodo
            # corrente (= a chi sta per muovere). Verifica invariante.
            assert env_attivo.bot_color == node.player_to_move, (
                f"Invariante rotta: env_attivo.bot_color={env_attivo.bot_color}, "
                f"node.player_to_move={node.player_to_move}"
            )

            # Esegui l'azione, eventualmente switchando env se finisce il turno
            env_attivo.step(child.action_taken)
            env_attivo = _switch_se_turno_finito(envs, env_attivo)

            # Post-step: env_attivo deve corrispondere al player del child
            # (eccetto se il child e' terminale: in quel caso giocatore_corrente
            # non cambia, env_attivo resta quello del parent, e il child ha
            # player_to_move = parent.player_to_move).
            if not env_attivo.stato.terminata:
                assert env_attivo.bot_color == child.player_to_move, (
                    f"Mismatch dopo step+switch: env={env_attivo.bot_color}, "
                    f"child={child.player_to_move}"
                )

            node = child
            path.append(node)

        # ════════════════════════════════════════════════════════════
        # 2. EXPANSION + EVALUATION
        # ════════════════════════════════════════════════════════════
        if node.is_terminal:
            # Foglia terminale: usa il valore terminale gia' calcolato
            value = node.terminal_value
            leaf_player = node.player_to_move
        else:
            # Foglia non terminale: chiama la rete UNA VOLTA
            # IMPORTANTE: env_attivo.bot_color == node.player_to_move per
            # invariante, quindi observation e' orientata sul leaf player.
            obs = env_attivo._costruisci_observation()
            info = env_attivo._costruisci_info()
            mask_np = info["action_mask"]

            # Sposta tensori sul device della rete
            net_device = next(net.parameters()).device
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(net_device)
            mask_t = torch.from_numpy(mask_np).bool().to(net_device)

            with torch.no_grad():
                policy_logits, value_tensor = net(obs_t)
                policy_dist = apply_mask_and_softmax(policy_logits[0], mask_t)

            value = float(value_tensor.item())  # dal POV del leaf player
            policy_dist_cpu = policy_dist.cpu()

            # Espandi: crea TUTTI i figli per le azioni legali
            legal_actions = np.where(mask_np)[0].tolist()
            node.legal_actions = legal_actions

            # Snapshot del nodo (parent dei figli) se non gia' salvato.
            # Compatibile con envs[node.player_to_move] (= env_attivo).
            if node.snapshot is None:
                node.snapshot = env_attivo.snapshot()
            snap_node = node.snapshot
            snap_node_color = node.player_to_move  # invariante

            for action in legal_actions:
                # Restore al parent state (env_attivo dello snap_node)
                env_attivo = _restore_e_riallinea(envs, snap_node_color, snap_node)

                obs_new, reward, term, trunc, info_new = env_attivo.step(int(action))

                # Switch se il turno e' finito e partita continua
                if not (term or trunc) and not env_attivo.stato.terminata:
                    env_attivo = _switch_se_turno_finito(envs, env_attivo)

                # player_to_move del child:
                # - se non terminata: env_attivo.bot_color (corretto post-switch)
                # - se terminata: stato.giocatore_corrente (non cambia in
                #   termina_partita_per_*; e' il colore "logico" del child).
                if env_attivo.stato.terminata or term or trunc:
                    child_player = env_attivo.stato.giocatore_corrente
                else:
                    child_player = env_attivo.bot_color

                # Snapshot del child:
                # - terminale: None (non sara' mai espanso, non serve)
                # - non terminale: snapshot di env_attivo (compatibile con
                #   envs[child_player], invariante rispettata).
                if term or trunc or env_attivo.stato.terminata:
                    child_snapshot = None
                else:
                    child_snapshot = env_attivo.snapshot()

                child = Node(
                    P=float(policy_dist_cpu[action].item()),
                    parent=node,
                    action_taken=int(action),
                    player_to_move=child_player,
                    snapshot=child_snapshot,
                )

                if term or trunc or env_attivo.stato.terminata:
                    child.is_terminal = True
                    # terminal_value: e' il reward dell'ultimo step. Reward e'
                    # dal POV di env_attivo.bot_color (quello che ha fatto step,
                    # = parent.player_to_move = snap_node_color). Per il backup
                    # usiamo player_leaf=child.player_to_move, che potrebbe
                    # essere diverso. Quindi convertiamo: se child_player ==
                    # snap_node_color, va bene; altrimenti flip.
                    #
                    # In pratica in 1v1, su step terminale stato.giocatore_corrente
                    # NON cambia (avanza_turno saltato), quindi child_player ==
                    # snap_node_color sempre. Lasciamo l'assert per sicurezza.
                    assert child_player == snap_node_color, (
                        f"Caso non previsto: terminale con cambio giocatore "
                        f"(snap_node_color={snap_node_color}, child={child_player})"
                    )
                    child.terminal_value = float(reward)

                node.children[int(action)] = child

            # Restore al snap del nodo per coerenza prima del backup
            env_attivo = _restore_e_riallinea(envs, snap_node_color, snap_node)

            leaf_player = node.player_to_move

            # Warm-start opzionale (sovrascrive value della rete)
            if rollout_value_fn is not None:
                value = rollout_value_fn(env_attivo)

        # ════════════════════════════════════════════════════════════
        # 3. BACKUP — propaga il value lungo il path
        # ════════════════════════════════════════════════════════════
        backup(path, value, leaf_player)

    finally:
        # ════════════════════════════════════════════════════════════
        # REGOLA D'ORO: env restorato allo stato della root al termine.
        # Restoriamo sull'env del root.player_to_move e riallineiamo l'altro.
        # ════════════════════════════════════════════════════════════
        _restore_e_riallinea(envs, root.player_to_move, root.snapshot)
