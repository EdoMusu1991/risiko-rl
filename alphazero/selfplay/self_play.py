"""
self_play.py — Gioca una partita completa con MCTS+rete e raccoglie dati.

Sub-step 5b della Settimana 4.

NOTA SETTIMANA 6 — DUE FUNZIONI:
1. gioca_partita_selfplay (legacy): usa un solo env con bot_color=BLU. Gli
   altri colori vengono giocati dall'env in automatico (bot interno random/
   euristico). Quindi raccoglie sample SOLO da BLU. Mantenuta per A/B test
   e backward-compat.

2. gioca_partita_selfplay_simmetrica: usa DUE env templati (BLU e ROSSO) con
   stato e rng condivisi via re-aliasing al cambio turno. Entrambi i colori
   passano per MCTS+rete -> sample BLU > 0 E sample ROSSO > 0. Limitato a
   mode_1v1=True.

   - PR1 (Settimana 6): risolto self-play esterno. Raccolta sample
     simmetrica.
   - PR2 (Settimana 6): risolto MCTS interno tramite search_simmetrico /
     simulate_simmetrico, che usano DUE env anche dentro il rollout MCTS.
     Eliminato il bug "step() chiamato fuori turno bot" e il bug di segno
     latente sulla observation orientata su bot_color.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch

from risiko_env import RisikoEnv
from .node import Node
from .search import search, visite_to_policy_full
from ..network import RisikoNet, ACTION_DIM


TEMPERATURE_DROP_STEP = 30


@dataclass
class TrainingSample:
    obs: np.ndarray
    mask: np.ndarray
    policy_target: np.ndarray
    player_at_state: str
    value_target: float = 0.0


def gioca_partita_selfplay(
    env: RisikoEnv,
    net: RisikoNet,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    temperature_drop_step: int = TEMPERATURE_DROP_STEP,
    seed: Optional[int] = None,
    max_decisioni: int = 2000,
    verbose: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[List[TrainingSample], dict]:
    if rng is None:
        rng = np.random.default_rng(seed)
    
    obs, info = env.reset(seed=seed)
    samples: List[TrainingSample] = []
    n_decisioni = 0
    
    while True:
        if n_decisioni >= max_decisioni:
            break
        
        mask = info["action_mask"]
        n_legali = int(mask.sum())
        
        if n_legali == 0:
            break
        if n_legali == 1:
            azione = int(np.where(mask)[0][0])
            obs, reward, term, trunc, info = env.step(azione)
            if term or trunc:
                break
            continue
        
        player = env.stato.giocatore_corrente
        root = Node(
            snapshot=env.snapshot(),
            player_to_move=player,
            P=1.0,
        )
        
        T = 1.0 if n_decisioni < temperature_drop_step else 0.0
        
        action, _ = search(
            root, env, net,
            n_simulations=n_simulations,
            c_puct=c_puct,
            temperature=T,
            rng=rng,
        )
        
        policy_target = visite_to_policy_full(root, ACTION_DIM, temperature=T)
        
        samples.append(TrainingSample(
            obs=obs.copy(),
            mask=mask.copy(),
            policy_target=policy_target,
            player_at_state=player,
            value_target=0.0,
        ))
        
        obs, reward, term, trunc, info = env.step(action)
        n_decisioni += 1
        
        if verbose and n_decisioni % 50 == 0:
            print(f"    decisione {n_decisioni}, sotto_fase={env.sotto_fase}, "
                  f"player={env.stato.giocatore_corrente}")
        
        if term or trunc:
            break
    
    reward_finale = float(reward)
    bot_color = env.bot_color
    
    for sample in samples:
        if sample.player_at_state == bot_color:
            sample.value_target = reward_finale
        else:
            sample.value_target = -reward_finale
    
    stats = {
        "n_decisioni_mcts": n_decisioni,
        "vincitore": info.get("vincitore"),
        "motivo_fine": info.get("motivo_fine"),
        "reward_finale": reward_finale,
        "n_samples": len(samples),
    }
    
    return samples, stats


# ═════════════════════════════════════════════════════════════════════════
#  PR1 — SELF-PLAY SIMMETRICO (fix bug raccolta sample monocromatica)
# ═════════════════════════════════════════════════════════════════════════

def gioca_partita_selfplay_simmetrica(
    net,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    temperature_drop_step: int = TEMPERATURE_DROP_STEP,
    seed: Optional[int] = None,
    max_decisioni: int = 2000,
    verbose: bool = False,
    rng: Optional[np.random.Generator] = None,
    *,
    mode_1v1: bool = True,
    log_eventi: bool = False,
    max_steps: int = 5000,
    reward_mode: str = "binary",
    policy_fn=None,
) -> tuple[List[TrainingSample], dict]:
    """
    Self-play simmetrico per AlphaZero (PR1 fix simmetria).

    A differenza di gioca_partita_selfplay (un solo env, ROSSO giocato dal bot
    interno), qui usiamo DUE env "templati" — uno per BLU e uno per ROSSO —
    con stato e rng condivisi tramite re-aliasing al cambio turno. Entrambi
    hanno _skip_giro_avversari=True, quindi nessuno dei due fa giocare
    automaticamente i turni dell'altro.

    Risultato: ogni decisione del giocatore corrente passa per MCTS+rete (o
    per policy_fn, se passata). Ogni decisione genera un TrainingSample con
    player_at_state = colore di chi muoveva. Atteso: n_samples_blu > 0 E
    n_samples_rosso > 0.

    Limitato a mode_1v1=True. 4-player non e' in scope per PR1.

    NOTA: questa funzione NON corregge il bug "observation centrata su
    bot_color durante MCTS expansion". Quel bug rimane in simulate.py e va
    affrontato nel PR2. Qui l'observation salvata in ogni TrainingSample
    e' comunque corretta (orientata sul player corrente) perche' viene
    costruita da env_attivo, che e' allineato sul giocatore corrente.

    Args:
        net: rete RisikoNet (policy+value). Ignorato se policy_fn e' non None.
        n_simulations, c_puct, temperature_drop_step: parametri MCTS standard.
        seed: seed riproducibilita'.
        max_decisioni: failsafe contro loop infiniti.
        verbose: stampa progressi ogni 50 step.
        rng: np.random.Generator opzionale.
        mode_1v1, log_eventi, max_steps, reward_mode: forwardati a RisikoEnv.
        policy_fn: callable opzionale di firma
            policy_fn(env, obs, info, temperature, rng) -> (action_int, policy_target_array)
            Se passato, sostituisce MCTS+rete (usato dai test smoke per non
            dipendere da torch/rete reale).

    Returns:
        (samples, stats):
        samples: lista di TrainingSample. player_at_state in {"BLU", "ROSSO"}.
            value_target settato dopo la fine partita usando il reward finale
            dal POV del giocatore corrispondente.
        stats: dict con n_decisioni_totale, n_decisioni_mcts, vincitore,
            motivo_fine, reward_finale, ultimo_player, n_samples, n_samples_blu,
            n_samples_rosso, partita_terminata.
    """
    assert mode_1v1, (
        "PR1 supporta solo mode_1v1=True. Il caso 4-player verra' affrontato "
        "in un PR successivo (servono 4 env o un design diverso)."
    )

    if rng is None:
        rng = np.random.default_rng(seed)

    # ── Setup due env templati ──
    env_blu = RisikoEnv(
        bot_color="BLU",
        mode_1v1=True,
        seed=seed,
        log_eventi=log_eventi,
        max_steps=max_steps,
        reward_mode=reward_mode,
    )
    env_blu._skip_giro_avversari = True

    env_rosso = RisikoEnv(
        bot_color="ROSSO",
        mode_1v1=True,
        seed=None,  # NON resettato: stato e rng iniettati al primo cambio turno
        log_eventi=log_eventi,
        max_steps=max_steps,
        reward_mode=reward_mode,
    )
    env_rosso._skip_giro_avversari = True
    # NB: env_rosso non chiama mai reset(). Le sue strutture interne
    # (sotto_fase=None, step_count=0, _combinazioni_tris=[]) vengono "armate"
    # da _inizia_fase_tris() al cambio turno, dopo aver iniettato stato/rng.

    obs, info = env_blu.reset(seed=seed)
    assert env_blu.stato.giocatore_corrente == "BLU", (
        f"In mode_1v1, BLU dovrebbe muovere per primo. "
        f"Trovato: {env_blu.stato.giocatore_corrente}"
    )

    env_attivo = env_blu
    samples: List[TrainingSample] = []
    n_decisioni = 0
    n_decisioni_mcts = 0
    reward = 0.0
    term = False
    trunc = False

    while True:
        if n_decisioni >= max_decisioni:
            break
        if env_attivo.stato.terminata:
            break

        mask = info["action_mask"]
        n_legali = int(mask.sum())

        if n_legali == 0:
            # Nessuna azione legale: non dovrebbe succedere ma usciamo per
            # non bloccarci. Gli edge case (es. tutti i legali sono
            # forced-no-op) sono gia' gestiti dall'env.
            break

        player = env_attivo.stato.giocatore_corrente
        # Invariante critica del design simmetrico
        assert player == env_attivo.bot_color, (
            f"Invariante violata: env_attivo.bot_color={env_attivo.bot_color} "
            f"ma giocatore_corrente={player}. Probabile bug nel re-aliasing."
        )

        T = 1.0 if n_decisioni < temperature_drop_step else 0.0

        if n_legali == 1:
            # Forced move: nessun sample, nessun MCTS
            action = int(np.where(mask)[0][0])
        else:
            if policy_fn is not None:
                action, policy_target = policy_fn(env_attivo, obs, info, T, rng)
            else:
                # PR2: usa search_simmetrico con dict envs invece di search
                # con env singolo. simulate_simmetrico gestisce i turni
                # avversari come livelli MIN dell'albero (AlphaZero puro)
                # invece di delegarli al bot interno dell'env.
                from .search import search_simmetrico
                envs_mcts = {"BLU": env_blu, "ROSSO": env_rosso}
                root = Node(
                    snapshot=env_attivo.snapshot(),
                    player_to_move=player,
                    P=1.0,
                )
                action, _ = search_simmetrico(
                    root, envs_mcts, net,
                    n_simulations=n_simulations,
                    c_puct=c_puct,
                    temperature=T,
                    rng=rng,
                )
                policy_target = visite_to_policy_full(root, ACTION_DIM, temperature=T)

            samples.append(TrainingSample(
                obs=obs.copy(),
                mask=mask.copy(),
                policy_target=policy_target,
                player_at_state=player,
                value_target=0.0,  # set sotto, dopo fine partita
            ))
            n_decisioni_mcts += 1

        # Step sull'env attivo
        obs, reward, term, trunc, info = env_attivo.step(int(action))
        n_decisioni += 1

        if verbose and n_decisioni % 50 == 0:
            print(
                f"    [simmetrico] dec={n_decisioni} env={env_attivo.bot_color} "
                f"sotto={env_attivo.sotto_fase} corrente={env_attivo.stato.giocatore_corrente}"
            )

        if term or trunc:
            break

        # ── Cambio env se il turno appena giocato e' finito ──
        # Quando sotto_fase=None, env_attivo.step() ha gia':
        #   1) eseguito _fine_turno_bot (pesca + sdadata + avanza_turno)
        #   2) NON eseguito _avanza_fino_a_turno_bot (skip_giro_avversari=True)
        # Quindi giocatore_corrente e' gia' l'altro colore, e dobbiamo solo
        # innescare l'altro env.
        if env_attivo.sotto_fase is None:
            env_passivo = env_rosso if env_attivo is env_blu else env_blu

            # Re-aliasing: critico perche' restore() di MCTS riassegna
            # self.stato/self.rng come nuovi oggetti deepcopy/setstate.
            # env_passivo, "stantio" da prima dell'MCTS, deve riallinearsi.
            env_passivo.stato = env_attivo.stato
            env_passivo.rng = env_attivo.rng

            # Arma la prima sotto-fase (legge stato.giocatori[bot_color].carte)
            env_passivo._inizia_fase_tris()

            env_attivo = env_passivo
            obs = env_attivo._costruisci_observation()
            info = env_attivo._costruisci_info()

    # ── Calcolo value_target (AlphaZero style, antisimmetrico ±1) ──
    # IMPORTANTE: NON riusiamo _calcola_reward_finale dell'env per propagare
    # il value_target. _calcola_reward_finale e' "4-player aware" e usa
    # REWARD_PER_POSIZIONE che da' +0.3/-0.3 ai posti intermedi. In 1v1 con
    # vincitore=BLU questo darebbe reward_blu=+1.0 e reward_rosso=+0.3
    # (entrambi positivi!) — pessima loss per la rete value, che imparerebbe
    # che "stare li' a fine partita e' buono per chiunque".
    #
    # Per AlphaZero classico vogliamo invece un value antisimmetrico:
    #   vincitore -> +1, perdente -> -1, pareggio -> 0.
    # Il reward shaping intermedio (calcolato dentro step()) NON entra qui:
    # questi sono i value_target di FINE PARTITA, l'unica cosa che la rete
    # impara a stimare come "valore di uno stato".
    reward_finale = float(reward)  # solo per stats, non per i value_target
    ultimo_player = env_attivo.bot_color
    vincitore = env_attivo.stato.vincitore  # None se non terminata o pareggio
    partita_finita = bool(env_attivo.stato.terminata)

    def _value_target_per(colore: str) -> float:
        if not partita_finita:
            return 0.0  # truncated: nessun esito definito
        if vincitore is None:
            return 0.0  # pareggio (cap_sicurezza con punteggi pari)
        return 1.0 if vincitore == colore else -1.0

    for sample in samples:
        sample.value_target = _value_target_per(sample.player_at_state)

    n_blu = sum(1 for s in samples if s.player_at_state == "BLU")
    n_rosso = sum(1 for s in samples if s.player_at_state == "ROSSO")

    stats = {
        "n_decisioni_totale": n_decisioni,
        "n_decisioni_mcts": n_decisioni_mcts,
        "vincitore": info.get("vincitore"),
        "motivo_fine": info.get("motivo_fine"),
        "reward_finale": reward_finale,
        "ultimo_player": ultimo_player,
        "n_samples": len(samples),
        "n_samples_blu": n_blu,
        "n_samples_rosso": n_rosso,
        "partita_terminata": bool(term and not trunc),
        "truncated": bool(trunc),
    }

    return samples, stats
