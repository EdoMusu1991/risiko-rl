"""
parallel.py — Self-play parallelizzato (PR3).

Wrapper di multiprocessing per gioca_partita_selfplay_simmetrica (PR1+PR2).

Idea: 1 processo worker = 1 partita alla volta, in seriale, ma piu' worker
girano in parallelo su core diversi. La rete viene replicata in ogni worker
via state_dict serializzato. Niente modifiche al codice esistente di
simulate.py / search.py / self_play.py / env.py.

Uso tipico:
    samples_per_partita, stats_per_partita = gioca_n_partite_parallele(
        net=mia_rete,
        n_partite=8,
        n_worker=4,
        base_seed=42,
        n_simulations=20,
        max_decisioni=1500,
    )

Decisioni di design:
1. torch.set_num_threads(1) nei worker. Senza, N processi x M thread BLAS
   si combattono sui core disponibili e PR3 va PIU' LENTO del sequenziale.
   Stesso discorso per OMP/MKL via env vars.
2. Spawn invece di fork. PyTorch + fork() post-CUDA-init e' problematico
   (deadlock noti sul driver CUDA). Spawn paga ~1s di overhead per worker
   all'avvio, trascurabile su partite da 100s+.
3. Pool con initializer: la rete viene costruita 1 volta per worker, non
   a ogni partita. Lo state_dict (~2.7MB) viene serializzato 1 volta nel
   main e passato come bytes ai worker.
4. Pool nuovo per ogni chiamata di gioca_n_partite_parallele. Per
   AlphaZero generazionale (1 iterazione = 1 set di partite con rete fissa)
   e' la semantica corretta. Per training online si potra' fare un wrapper.
5. Seed per partita = base_seed + idx. Determinismo riproducibile a parita'
   di rete e numero di thread BLAS.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import io
import os
import multiprocessing as mp

import torch

from ..network import RisikoNet
from .self_play import TrainingSample, gioca_partita_selfplay_simmetrica


# ─────────────────────────────────────────────────────────────────
#  WORKER STATE (globali del processo worker, settate da _worker_init)
# ─────────────────────────────────────────────────────────────────

_worker_net: Optional[RisikoNet] = None


def _worker_init(state_dict_bytes: bytes, net_kwargs: dict) -> None:
    """
    Initializer chiamato 1 volta per worker (non per partita).

    Cosa fa:
    1. Forza single-thread per torch / OMP / MKL nel worker
       (altrimenti N worker x M thread BLAS = contesa CPU = NO speedup).
    2. Deserializza lo state_dict ricevuto dal main.
    3. Ricostruisce RisikoNet su CPU, carica i pesi, mette in eval().
    4. Salva la rete in una globale del processo per riuso tra partite.
    """
    global _worker_net

    # Single-thread per il worker.
    # NB: le env var vanno settate PRIMA che torch carichi BLAS, ma in pratica
    # in spawn il modulo torch viene re-importato qui quindi siamo a posto.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    state_dict = torch.load(
        io.BytesIO(state_dict_bytes),
        map_location="cpu",
        weights_only=True,
    )

    net = RisikoNet(**net_kwargs)
    net.load_state_dict(state_dict)
    net.eval()
    _worker_net = net


def _worker_play(args: tuple) -> tuple:
    """
    Una partita per chiamata. La rete e' gia' caricata in _worker_net.

    Args:
        args: (game_idx, seed, game_kwargs)

    Returns:
        (game_idx, samples, stats)
    """
    global _worker_net
    assert _worker_net is not None, "Worker non inizializzato (manca _worker_init)"

    game_idx, seed, game_kwargs = args

    samples, stats = gioca_partita_selfplay_simmetrica(
        net=_worker_net,
        seed=seed,
        **game_kwargs,
    )
    return (game_idx, samples, stats)


# ─────────────────────────────────────────────────────────────────
#  API PUBBLICA
# ─────────────────────────────────────────────────────────────────

def gioca_n_partite_parallele(
    net: RisikoNet,
    n_partite: int,
    n_worker: int,
    base_seed: int = 0,
    *,
    net_kwargs: Optional[dict] = None,
    start_method: str = "spawn",
    **game_kwargs,
) -> Tuple[List[List[TrainingSample]], List[dict]]:
    """
    Gioca n_partite in parallelo su n_worker processi.

    Ogni worker riceve uno snapshot della rete via state_dict, lo carica
    su CPU in eval mode, e gioca le partite assegnate dal Pool. Le
    partite hanno seed = base_seed + i per i in [0, n_partite).

    Args:
        net: RisikoNet di cui usare i pesi correnti. Verra' copiata nei worker.
            La rete originale non viene modificata.
        n_partite: numero totale di partite da giocare.
        n_worker: numero di processi worker da lanciare. Se > n_partite,
            viene clampato a n_partite (no senso avere worker idle).
            Tipico: n_cpu_fisici - 1 (lasciando 1 core al main).
        base_seed: seed di base. Partita i usera' seed=base_seed+i.
        net_kwargs: dict di kwargs per costruire RisikoNet nei worker.
            Default {} = costruttore con argomenti default. Necessario solo
            se la rete originale e' stata customizzata su input/action dim.
        start_method: 'spawn' (default, sicuro), 'fork' (piu' veloce ma
            problematico con CUDA), 'forkserver'. Vedi multiprocessing docs.
        **game_kwargs: argomenti forwardati a gioca_partita_selfplay_simmetrica.
            Es: n_simulations, c_puct, max_decisioni, max_steps, reward_mode.
            NB: NON passare seed qui (e' calcolato dal base_seed) ne' net.

    Returns:
        (samples_list, stats_list):
            samples_list[i] = lista di TrainingSample della partita i-esima
                (seed = base_seed + i).
            stats_list[i] = dict di statistiche della partita i-esima.
        L'ordine e' quello di game_idx, indipendente dall'ordine in cui i
        worker hanno completato.

    Esempio:
        >>> net = RisikoNet()
        >>> samples, stats = gioca_n_partite_parallele(
        ...     net, n_partite=8, n_worker=4, base_seed=42,
        ...     n_simulations=20, max_decisioni=1500,
        ... )
        >>> tot_samples = sum(len(s) for s in samples)
        >>> n_vinte = sum(1 for st in stats if st["vincitore"] is not None)
    """
    if n_partite <= 0:
        raise ValueError(f"n_partite deve essere > 0, ricevuto {n_partite}")
    if n_worker <= 0:
        raise ValueError(f"n_worker deve essere > 0, ricevuto {n_worker}")
    if n_worker > n_partite:
        n_worker = n_partite

    # Validazione game_kwargs: 'seed' e' gestito internamente (= base_seed + idx)
    if "seed" in game_kwargs:
        raise ValueError(
            "Non passare 'seed' in game_kwargs. Usa base_seed (la partita i-esima "
            "usera' base_seed + i)."
        )

    if net_kwargs is None:
        net_kwargs = {}

    # Serializza state_dict una volta sola
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    state_dict_bytes = buf.getvalue()

    # Job list: (game_idx, seed, game_kwargs_copy)
    # Ogni dict e' una copia per evitare condivisione tra processi
    jobs = [
        (i, base_seed + i, dict(game_kwargs))
        for i in range(n_partite)
    ]

    ctx = mp.get_context(start_method)

    with ctx.Pool(
        processes=n_worker,
        initializer=_worker_init,
        initargs=(state_dict_bytes, net_kwargs),
    ) as pool:
        # pool.map preserva l'ordine, ma per robustezza ordiniamo per game_idx
        results = pool.map(_worker_play, jobs)

    results.sort(key=lambda r: r[0])

    samples_list = [r[1] for r in results]
    stats_list = [r[2] for r in results]

    return samples_list, stats_list


# ─────────────────────────────────────────────────────────────────
#  EVAL: net (BLU) vs bot interno (default random) — anche parallelo
# ─────────────────────────────────────────────────────────────────

# Worker globals (riusiamo il pattern dei selfplay worker)
_eval_worker_net: Optional[RisikoNet] = None


def _eval_worker_init(state_dict_bytes: bytes, net_kwargs: dict) -> None:
    """
    Initializer per worker di valutazione.
    Stesso del self-play: forza single-thread, ricostruisce la rete.
    """
    global _eval_worker_net

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    state_dict = torch.load(
        io.BytesIO(state_dict_bytes),
        map_location="cpu",
        weights_only=True,
    )

    net = RisikoNet(**net_kwargs)
    net.load_state_dict(state_dict)
    net.eval()
    _eval_worker_net = net


def _eval_worker_play_vs_random(args: tuple) -> tuple:
    """
    Una partita: rete (giocatore = bot_color) vs bot interno per gli altri.

    In mode_1v1 + bot_color="BLU" + NON _skip_giro_avversari, l'env fa
    giocare automaticamente ROSSO con bot_random ad ogni cambio turno.
    Quindi il "nostro" bot affronta un random.

    Usa gioca_partita_selfplay (legacy, non simmetrica): MCTS espande solo
    nodi BLU, il bug PR2 sull'observation orientata su bot_color non si
    manifesta perche' ROSSO non viene mai espanso in MCTS (lo gioca l'env).

    Args:
        args: (game_idx, seed, bot_color, n_simulations, max_decisioni, c_puct)

    Returns:
        (game_idx, stats)
    """
    global _eval_worker_net
    assert _eval_worker_net is not None, "Worker non inizializzato"

    game_idx, seed, bot_color, n_simulations, max_decisioni, c_puct = args

    # Import lazy: questi moduli vengono importati per la prima volta nel worker
    from risiko_env import RisikoEnv
    from .self_play import gioca_partita_selfplay

    env = RisikoEnv(bot_color=bot_color, mode_1v1=True, seed=seed)
    _, stats = gioca_partita_selfplay(
        env=env,
        net=_eval_worker_net,
        n_simulations=n_simulations,
        c_puct=c_puct,
        seed=seed,
        max_decisioni=max_decisioni,
    )
    return (game_idx, stats)


def gioca_n_partite_vs_random_parallele(
    net: RisikoNet,
    n_partite: int,
    n_worker: int,
    base_seed: int = 0,
    *,
    bot_color: str = "BLU",
    n_simulations: int = 10,
    max_decisioni: int = 1500,
    c_puct: float = 1.5,
    net_kwargs: Optional[dict] = None,
    start_method: str = "spawn",
) -> List[dict]:
    """
    Gioca n_partite "rete vs bot interno" in parallelo.

    Per ogni partita: la rete gioca col colore `bot_color` (default BLU),
    e l'altro colore (ROSSO in mode_1v1) e' giocato dal bot random
    interno dell'env. Util per valutare una rete contro baseline random.

    Args:
        net: RisikoNet con cui giocare.
        n_partite: numero totale di partite.
        n_worker: processi worker.
        base_seed: seed di base. Partita i usa seed=base_seed+i.
        bot_color: colore controllato dalla rete ("BLU" o "ROSSO").
        n_simulations, max_decisioni, c_puct: parametri MCTS.
        net_kwargs: kwargs per costruire RisikoNet nei worker (default {}).
        start_method: 'spawn' (default), 'fork', 'forkserver'.

    Returns:
        Lista di n_partite dict di statistiche, ordinata per game_idx.
        Ogni stats ha 'vincitore' che e' "BLU"/"ROSSO"/None.
    """
    if n_partite <= 0:
        raise ValueError(f"n_partite deve essere > 0, ricevuto {n_partite}")
    if n_worker <= 0:
        raise ValueError(f"n_worker deve essere > 0, ricevuto {n_worker}")
    if n_worker > n_partite:
        n_worker = n_partite
    if bot_color not in ("BLU", "ROSSO"):
        raise ValueError(f"bot_color deve essere BLU o ROSSO, ricevuto {bot_color}")

    if net_kwargs is None:
        net_kwargs = {}

    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    state_dict_bytes = buf.getvalue()

    jobs = [
        (i, base_seed + i, bot_color, n_simulations, max_decisioni, c_puct)
        for i in range(n_partite)
    ]

    ctx = mp.get_context(start_method)

    with ctx.Pool(
        processes=n_worker,
        initializer=_eval_worker_init,
        initargs=(state_dict_bytes, net_kwargs),
    ) as pool:
        results = pool.map(_eval_worker_play_vs_random, jobs)

    results.sort(key=lambda r: r[0])
    return [r[1] for r in results]