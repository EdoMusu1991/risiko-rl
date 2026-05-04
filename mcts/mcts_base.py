"""
mcts_base.py — MCTS base con UCB e rollout euristico/random.

Implementazione minimale per Settimana 2 (Mese 1) del progetto AlphaZero.
NON include rete neurale: usa rollout policy come stima del valore.

Architettura:
- Node con visite (N), valore cumulativo (W), prior (P) uniforme
- Selection con UCB1 (PUCT senza prior informativo)
- Expansion lazy: crea figli solo quando il nodo viene visitato
- Simulation: rollout euristico o random fino a fine partita
- Backup: media valori lungo il path

Decisione tecnica:
- Lavoriamo a livello di azioni dell'env (action space gerarchico Risiko)
- Non separiamo fase tris/rinforzo/attacco/spostamento: MCTS sceglie
  la prossima azione legale qualunque sia la fase
- Snapshot/restore per ogni simulazione (verificato perf < 1ms)

Limiti noti:
- No prior informativo (sara' la rete in fase 2)
- No multi-thread (puramente sequenziale)
- No transposition table (ricompiamo nodi che potrebbero essere uguali)
"""

import math
import random
import time
from typing import Optional, Callable

import numpy as np


# ─────────────────────────────────────────────────────────────────────
#  COSTANTI
# ─────────────────────────────────────────────────────────────────────

# UCB exploration constant. Standard AlphaZero usa 1.0-2.0. Iniziamo con 1.5.
C_UCB = 1.5

# Limite di profondita' per i rollout (oltre questo, valutiamo da margine punti).
# Risiko 1v1 dura ~36-40 round = ~70-80 turni totali = ~250-400 step.
# Mettiamo cap generoso ma esistente per evitare runaway.
MAX_ROLLOUT_DEPTH = 100


# ─────────────────────────────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────────────────────────────

class MCTSNode:
    """
    Nodo dell'albero MCTS. Rappresenta uno stato dell'env.

    Attributi:
        N: numero di visite
        W: somma dei valori dalle simulazioni
        P: prior probability (uniforme = 1.0 / n_legali nel prototipo)
        children: dict {azione_int -> MCTSNode}
        legali: lista azioni legali da questo stato
        is_terminal: True se la partita e' finita in questo stato
        terminal_value: valore se terminale (-1 a +1)
    """

    __slots__ = ("N", "W", "P", "children", "legali", "is_terminal",
                 "terminal_value", "snapshot")

    def __init__(self, P: float = 1.0):
        self.N = 0
        self.W = 0.0
        self.P = P
        self.children: dict = {}
        self.legali: Optional[list] = None
        self.is_terminal = False
        self.terminal_value = 0.0
        self.snapshot = None  # popolato lazy

    @property
    def Q(self) -> float:
        """Valore medio (W/N), 0 se non visitato."""
        return self.W / self.N if self.N > 0 else 0.0

    def is_expanded(self) -> bool:
        return self.legali is not None


# ─────────────────────────────────────────────────────────────────────
#  ROLLOUT POLICIES
# ─────────────────────────────────────────────────────────────────────

def rollout_random(env, max_depth: int = MAX_ROLLOUT_DEPTH) -> float:
    """
    Rollout completamente random fino a fine partita.
    Ritorna il reward dal punto di vista del bot_color dell'env.
    """
    info = env._costruisci_info()
    rng = env.rng  # usa lo stesso RNG dell'env per riproducibilita'

    for _ in range(max_depth):
        if info is None:
            break
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        if len(legali) == 0:
            break
        action = int(rng.choice(legali))
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            return float(reward)

    # Max depth raggiunto: stima da margine punti
    return env._calcola_reward_margin()


def rollout_euristico(env, max_depth: int = MAX_ROLLOUT_DEPTH) -> float:
    """
    Rollout con bot euristico. Per la fase del bot_color, gioca random
    (il MCTS sceglie le sue azioni); per gli altri, gioca euristico.

    Ma in modalita' MCTS rollout, "il bot" e' simulato anche lui — non c'e'
    un agente esterno che lo guida. Quindi qui usiamo l'euristico per
    TUTTI i giocatori (incluso bot_color) durante il rollout.

    Implementazione: forziamo l'env a usare bot euristici per gli avversari,
    e per il bot_color giochiamo le azioni euristiche tramite step() random
    (no, dobbiamo bypassare e usare gioca_turno_euristico direttamente).

    SOLUZIONE PIU' SEMPLICE: durante il rollout, sostituiamo gli avversari
    custom con "euristico" e per il bot_color facciamo step random ma
    filtrato per somigliare all'euristico (=> hard).

    SOLUZIONE EFFETTIVA: durante rollout, usiamo l'env "raw" senza Gym,
    chiamando gioca_turno_euristico per tutti i giocatori vivi.
    """
    from risiko_env.bot_euristico import gioca_turno_euristico
    from risiko_env.motore import avanza_turno
    from risiko_env.sdadata import gestisci_fine_turno

    stato = env.stato
    rng = env.rng
    bot_color = env.bot_color

    for _ in range(max_depth):
        if stato.terminata:
            break
        c = stato.giocatore_corrente
        if not stato.giocatori[c].vivo:
            avanza_turno(stato)
            continue
        if not stato.territori_di(c):
            stato.giocatori[c].vivo = False
            avanza_turno(stato)
            continue
        # Tutti giocano euristico durante rollout (anche il bot_color)
        gioca_turno_euristico(stato, c, rng)
        if stato.terminata:
            break
        gestisci_fine_turno(stato, c, rng)
        if stato.terminata:
            break
        avanza_turno(stato)

    # Calcola reward dal punto di vista del bot_color
    if stato.vincitore == bot_color:
        return 1.0
    if not stato.giocatori[bot_color].vivo:
        return -1.0
    # Sdadata o max_depth: usa margine punti
    return env._calcola_reward_margin()


# ─────────────────────────────────────────────────────────────────────
#  MCTS CORE
# ─────────────────────────────────────────────────────────────────────

class MCTS:
    """
    Monte Carlo Tree Search con UCB1 e rollout policy.

    Uso:
        mcts = MCTS(env, n_simulations=100)
        action = mcts.search()
    """

    def __init__(
        self,
        env,
        n_simulations: int = 100,
        rollout_policy: str = "euristico",  # "euristico" o "random"
        c_ucb: float = C_UCB,
        max_rollout_depth: int = MAX_ROLLOUT_DEPTH,
        time_budget_sec: Optional[float] = None,
    ):
        """
        Args:
            env: RisikoEnv (deve supportare snapshot/restore)
            n_simulations: numero massimo di simulazioni per mossa
            rollout_policy: "euristico" o "random"
            c_ucb: costante UCB exploration
            max_rollout_depth: profondita' massima del rollout
            time_budget_sec: se settato, ferma le simulazioni al raggiungimento
                             del budget tempo (anche prima di n_simulations)
        """
        self.env = env
        self.n_simulations = n_simulations
        self.c_ucb = c_ucb
        self.max_rollout_depth = max_rollout_depth
        self.time_budget_sec = time_budget_sec

        if rollout_policy == "euristico":
            self.rollout_fn = rollout_euristico
        elif rollout_policy == "random":
            self.rollout_fn = rollout_random
        else:
            raise ValueError(f"rollout_policy invalido: {rollout_policy}")

    def search(self) -> int:
        """
        Esegue MCTS dallo stato corrente e ritorna l'azione migliore.
        L'env DEVE essere nel turno del bot e in stato non terminale.
        """
        # Salva snapshot dell'env reale (non lo tocchiamo durante la search)
        snap_root = self.env.snapshot()

        # Crea root
        root = MCTSNode(P=1.0)
        root.snapshot = snap_root
        info = self.env._costruisci_info()
        legali_root = list(np.where(info["action_mask"])[0])
        if not legali_root:
            return 0  # nessuna azione legale (non dovrebbe succedere)
        root.legali = legali_root
        prior = 1.0 / len(legali_root)
        for a in legali_root:
            root.children[a] = MCTSNode(P=prior)

        # Caso speciale: una sola azione legale, no MCTS
        if len(legali_root) == 1:
            self.env.restore(snap_root)
            self.last_search_stats = {
                "n_simulations": 0,
                "time_sec": 0.0,
                "single_legal_action": True,
                "best_action": legali_root[0],
            }
            return legali_root[0]

        # Cicli di simulazione
        t_start = time.perf_counter()
        n_done = 0
        while n_done < self.n_simulations:
            if self.time_budget_sec is not None:
                if time.perf_counter() - t_start > self.time_budget_sec:
                    break
            self._simulate(root)
            n_done += 1

        # Ripristina env reale
        self.env.restore(snap_root)

        # Scegli azione con piu' visite alla root
        best_action = max(root.children.items(), key=lambda kv: kv[1].N)[0]

        # Memorizza statistiche per debugging
        self.last_search_stats = {
            "n_simulations": n_done,
            "time_sec": time.perf_counter() - t_start,
            "root_q": root.Q,
            "children_visits": {a: c.N for a, c in root.children.items()},
            "children_q": {a: c.Q for a, c in root.children.items() if c.N > 0},
            "best_action": best_action,
            "best_visits": root.children[best_action].N,
        }

        return int(best_action)

    def _simulate(self, root: MCTSNode) -> None:
        """
        Esegue una simulazione MCTS dalla root:
        Selection -> Expansion -> Rollout -> Backup
        """
        # Restore env alla root
        self.env.restore(root.snapshot)

        path = [root]
        node = root

        # === SELECTION ===
        # Scendi finche' arrivi a un nodo non espanso o terminale
        while node.is_expanded() and not node.is_terminal:
            best_child = self._select_child(node)
            if best_child is None:
                break
            best_action = best_child[0]
            child = best_child[1]
            # Esegui l'azione
            obs, reward, term, trunc, info = self.env.step(int(best_action))
            path.append(child)
            node = child
            if term or trunc:
                node.is_terminal = True
                node.terminal_value = float(reward)
                break

        # === EXPANSION ===
        # Se non terminale e non ancora espanso, espandi
        if not node.is_terminal and not node.is_expanded():
            info = self.env._costruisci_info()
            mask = info["action_mask"]
            legali = list(np.where(mask)[0])
            if not legali:
                # Nessuna mossa legale: tratta come terminale
                node.is_terminal = True
                node.terminal_value = self.env._calcola_reward_margin()
            else:
                node.legali = legali
                node.snapshot = self.env.snapshot()
                prior = 1.0 / len(legali)
                for a in legali:
                    node.children[a] = MCTSNode(P=prior)

        # === ROLLOUT ===
        if node.is_terminal:
            value = node.terminal_value
        else:
            # Rollout dalla posizione corrente
            value = self.rollout_fn(self.env, self.max_rollout_depth)

        # === BACKUP ===
        # Propaga il valore lungo il path
        # NB: in giochi multi-player il valore va negato per ogni "turno
        # avversario", ma qui lavoriamo dal punto di vista FISSO del bot_color
        # (il rollout ritorna sempre il valore per bot_color), quindi non serve.
        for n in path:
            n.N += 1
            n.W += value

    def _select_child(self, node: MCTSNode) -> Optional[tuple]:
        """
        Seleziona il figlio con UCB massimo.
        UCB(child) = Q(child) + c * P(child) * sqrt(N_parent) / (1 + N_child)
        """
        if not node.children:
            return None

        sum_visits = sum(c.N for c in node.children.values())
        sqrt_sum = math.sqrt(max(1, sum_visits))

        best_score = -float("inf")
        best_pair = None
        for action, child in node.children.items():
            ucb = child.Q + self.c_ucb * child.P * sqrt_sum / (1 + child.N)
            if ucb > best_score:
                best_score = ucb
                best_pair = (action, child)

        return best_pair


# ─────────────────────────────────────────────────────────────────────
#  AGENT WRAPPER
# ─────────────────────────────────────────────────────────────────────

class MCTSAgent:
    """
    Wrapper per usare MCTS come agente in un loop di gioco.
    Compatibile con il pattern di valutazione standard (agent.predict).

    Ottimizzazione: in alcune sotto-fasi (rinforzo, spostamento, ecc.) MCTS
    e' computazionalmente costoso ma poco efficace (le azioni sono spesso
    quasi-equivalenti). Per default attiviamo MCTS solo nelle fasi "decisive":
    attacco, continua, quantita_conquista.

    Nelle altre fasi usiamo l'euristico (rinforzi mirati, ecc.).
    Questo riduce drasticamente il tempo per partita senza perdere troppo
    di qualita' decisionale (le scelte tattiche sono nelle fasi attivate).
    """

    # Sotto-fasi dove MCTS si attiva. Le altre usano l'euristico.
    SOTTOFASI_MCTS = {"attacco", "continua", "quantita_conquista"}

    def __init__(self, env, n_simulations: int = 100,
                 rollout_policy: str = "euristico",
                 time_budget_sec: Optional[float] = None,
                 mcts_only_phases: Optional[set] = None):
        """
        Args:
            mcts_only_phases: set di sotto_fase dove MCTS si attiva.
                Default: {"attacco", "continua", "quantita_conquista"}.
                Per disabilitare il filtraggio (MCTS sempre): set("all").
        """
        self.env = env
        self.n_simulations = n_simulations
        self.rollout_policy = rollout_policy
        self.time_budget_sec = time_budget_sec
        if mcts_only_phases is None:
            self.mcts_only_phases = self.SOTTOFASI_MCTS
        elif mcts_only_phases == "all":
            self.mcts_only_phases = None  # None = sempre MCTS
        else:
            self.mcts_only_phases = mcts_only_phases
        self._last_stats = None

    def predict(self, obs=None, action_masks=None, deterministic=True):
        """
        Ritorna (azione, None).
        Se la sotto-fase corrente non e' fra quelle MCTS, usa fallback rapido.
        """
        # Fast path: skip MCTS se sotto-fase banale
        if self.mcts_only_phases is not None and self.env.sotto_fase not in self.mcts_only_phases:
            action = self._fallback_euristico(action_masks)
            self._last_stats = {"fast_path": True, "sotto_fase": self.env.sotto_fase}
            return action, None

        # Fast path: una sola azione legale
        if action_masks is not None and int(action_masks.sum()) == 1:
            action = int(np.where(action_masks)[0][0])
            self._last_stats = {"single_legal": True}
            return action, None

        mcts = MCTS(
            self.env,
            n_simulations=self.n_simulations,
            rollout_policy=self.rollout_policy,
            time_budget_sec=self.time_budget_sec,
        )
        action = mcts.search()
        self._last_stats = mcts.last_search_stats
        return action, None

    def _fallback_euristico(self, action_masks) -> int:
        """
        Fallback rapido per fasi banali: usa l'euristico per scegliere.
        Per semplicita' sceglie azione random fra le legali.
        Future ottimizzazione: implementare logica euristica per fase.
        """
        legali = np.where(action_masks)[0]
        if len(legali) == 0:
            return 0
        return int(self.env.rng.choice(legali))
