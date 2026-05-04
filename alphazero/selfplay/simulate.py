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
