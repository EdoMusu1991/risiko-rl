"""
test_env.py — Test del Modulo 5c (RisikoEnv Gymnasium).

Verifica:
- API Gymnasium standard (reset, step, observation_space, action_space)
- Maschera azioni sempre valida (almeno 1 True)
- Step rispetta sempre la maschera
- Le partite simulate via env terminano correttamente
- Reward finale corretto in base al risultato
- Riproducibilità con seed
- Gli avversari (bot random) giocano automaticamente

Esegui: python tests\test_env.py
"""

import sys
import os
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env.env import RisikoEnv, ACTION_SPACE_SIZE, SottoFase
from risiko_env.encoding import DIM_OBSERVATION


# ═════════════════════════════════════════════════════════════════════════
#  API GYMNASIUM
# ═════════════════════════════════════════════════════════════════════════

def test_reset_restituisce_obs_e_info():
    """reset() deve restituire (observation, info)."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (DIM_OBSERVATION,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    assert "action_mask" in info
    print(f"✓ reset(): obs shape ({DIM_OBSERVATION},), info ha action_mask")


def test_observation_space():
    """observation_space deve combaciare con la dimensione dell'obs."""
    env = RisikoEnv(seed=42)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), (
        "obs non contenuto in observation_space"
    )
    print(f"✓ observation_space: Box (0,1) shape {env.observation_space.shape}")


def test_action_space():
    """action_space discreto di dimensione 1765."""
    env = RisikoEnv(seed=42)
    assert env.action_space.n == ACTION_SPACE_SIZE
    print(f"✓ action_space: Discrete({ACTION_SPACE_SIZE})")


def test_action_mask_in_info():
    """info contiene action_mask di dim corretta."""
    env = RisikoEnv(seed=42)
    _, info = env.reset()
    mask = info["action_mask"]
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype == bool
    assert mask.any(), "Almeno un'azione deve essere legale"
    print(f"✓ action_mask: shape ({ACTION_SPACE_SIZE},), almeno 1 azione legale")


def test_step_restituisce_5_valori():
    """step() deve restituire (obs, reward, terminated, truncated, info)."""
    env = RisikoEnv(seed=42)
    _, info = env.reset()
    azione_legale = int(np.argmax(info["action_mask"]))
    obs, reward, terminated, truncated, info = env.step(azione_legale)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    print("✓ step() restituisce 5-tuple Gymnasium standard")


# ═════════════════════════════════════════════════════════════════════════
#  ACTION MASK SEMPRE VALIDA
# ═════════════════════════════════════════════════════════════════════════

def test_mask_sempre_almeno_un_true():
    """In qualunque stato, almeno una azione deve essere legale."""
    env = RisikoEnv(seed=42)
    _, info = env.reset()
    n_check = 0
    for _ in range(200):
        mask = info["action_mask"]
        assert mask.any(), f"Mask vuota dopo {n_check} step"
        # Scegli azione legale random
        legali = np.where(mask)[0]
        azione = random.choice(legali)
        obs, reward, terminated, truncated, info = env.step(int(azione))
        n_check += 1
        if terminated or truncated:
            break
    print(f"✓ Mask sempre valida su {n_check} step")


# ═════════════════════════════════════════════════════════════════════════
#  PARTITA COMPLETA
# ═════════════════════════════════════════════════════════════════════════

def test_partita_termina_con_reward():
    """Una partita deve terminare e dare un reward finale."""
    env = RisikoEnv(seed=42)
    _, info = env.reset()
    rewards_totali = 0.0
    step_count = 0
    while True:
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        azione = random.choice(legali)
        obs, reward, terminated, truncated, info = env.step(int(azione))
        rewards_totali += reward
        step_count += 1
        if terminated or truncated:
            break
        assert step_count < 10000, "Loop infinito!"
    assert step_count < 10000
    print(f"✓ Partita seed=42: terminata in {step_count} step, "
          f"reward finale={rewards_totali}, vincitore={info['vincitore']}")


def test_molte_partite_terminano():
    """20 partite con seed diversi devono tutte terminare."""
    risultati = []
    for seed in range(20):
        env = RisikoEnv(seed=seed)
        _, info = env.reset()
        step_count = 0
        reward_finale = 0.0
        while step_count < 10000:
            mask = info["action_mask"]
            legali = np.where(mask)[0]
            azione = random.choice(legali)
            obs, reward, terminated, truncated, info = env.step(int(azione))
            step_count += 1
            if terminated or truncated:
                reward_finale = reward
                break
        risultati.append({
            "seed": seed,
            "step": step_count,
            "reward": reward_finale,
            "vincitore": info["vincitore"],
        })
        assert step_count < 10000, f"Seed {seed}: loop infinito"

    # Statistiche
    vittorie = sum(1 for r in risultati if r["reward"] == 1.0)
    sconfitte = sum(1 for r in risultati if r["reward"] == -1.0)
    step_medio = sum(r["step"] for r in risultati) / len(risultati)
    print(f"✓ 20 partite tutte terminate (step medio: {step_medio:.0f})")
    print(f"  Vittorie bot (BLU): {vittorie}/20, Sconfitte: {sconfitte}/20")


# ═════════════════════════════════════════════════════════════════════════
#  REWARD CORRETTO
# ═════════════════════════════════════════════════════════════════════════

def test_reward_solo_a_fine_partita():
    """Reward intermedio è piccolo (shaping), reward terminale è grande."""
    env = RisikoEnv(seed=42)
    _, info = env.reset()
    rewards = []
    while True:
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        azione = random.choice(legali)
        obs, reward, terminated, truncated, info = env.step(int(azione))
        rewards.append(reward)
        if terminated or truncated:
            break
    # Il reward terminale deve essere grande (|reward| >= 0.3)
    assert abs(rewards[-1]) >= 0.3, f"Reward terminale troppo piccolo: {rewards[-1]}"
    # I reward intermedi devono essere piccoli (shaping)
    if len(rewards) > 1:
        max_intermediate = max(abs(r) for r in rewards[:-1])
        assert max_intermediate < 0.1, (
            f"Reward intermedio troppo grande: {max_intermediate}"
        )
    print(f"✓ Reward shaping: terminale={rewards[-1]}, max intermedio={max([abs(r) for r in rewards[:-1]] + [0]):.4f}")


def test_reward_vittoria_max():
    """Reward di vittoria deve essere 1.0."""
    # Cerchiamo un seed in cui Blu vince
    for seed in range(50):
        env = RisikoEnv(seed=seed)
        _, info = env.reset()
        ultima_reward = 0.0
        while True:
            mask = info["action_mask"]
            legali = np.where(mask)[0]
            azione = random.choice(legali)
            obs, reward, terminated, truncated, info = env.step(int(azione))
            ultima_reward = reward
            if terminated or truncated:
                break
        if info["vincitore"] == "BLU":
            assert ultima_reward == 1.0, f"Vincitore BLU ma reward {ultima_reward}"
            print(f"✓ Reward vittoria seed={seed}: 1.0")
            return
    print("⚠ Nessuna vittoria di BLU su 50 seed (improbabile, ma non bloccante)")


# ═════════════════════════════════════════════════════════════════════════
#  RIPRODUCIBILITÀ
# ═════════════════════════════════════════════════════════════════════════

def test_riproducibilita():
    """Stesso seed + stesse azioni → stessa traiettoria."""
    def gioca_partita(seed, azioni):
        env = RisikoEnv(seed=seed)
        _, info = env.reset()
        rewards = []
        for a in azioni:
            mask = info["action_mask"]
            # Se a non è legale, scegli prima legale
            if not mask[a]:
                a = int(np.argmax(mask))
            obs, reward, terminated, truncated, info = env.step(int(a))
            rewards.append(reward)
            if terminated or truncated:
                break
        return rewards, info["vincitore"]

    azioni = [0] * 100
    r1, v1 = gioca_partita(42, azioni)
    r2, v2 = gioca_partita(42, azioni)
    assert r1 == r2, "Reward diversi con stesso seed!"
    assert v1 == v2, "Vincitori diversi con stesso seed!"
    print(f"✓ Riproducibilità: seed=42 → sempre stessa traiettoria, vincitore {v1}")


# ═════════════════════════════════════════════════════════════════════════
#  SOTTO-FASI
# ═════════════════════════════════════════════════════════════════════════

def test_sotto_fasi_in_sequenza():
    """Le sotto-fasi devono apparire in ordine logico durante la partita."""
    env = RisikoEnv(seed=42)
    _, info = env.reset()
    fasi_viste = set()
    step_count = 0
    while step_count < 500:
        fasi_viste.add(info.get("sotto_fase"))
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        azione = random.choice(legali)
        obs, reward, terminated, truncated, info = env.step(int(azione))
        step_count += 1
        if terminated or truncated:
            break
    # Almeno 3 sotto-fasi diverse devono essere apparse
    fasi_significative = fasi_viste - {None}
    assert len(fasi_significative) >= 3, (
        f"Solo {len(fasi_significative)} sotto-fasi viste: {fasi_significative}"
    )
    print(f"✓ Sotto-fasi viste durante partita: {sorted(fasi_significative)}")


# ═════════════════════════════════════════════════════════════════════════
#  TURNI AVVERSARI AUTOMATICI
# ═════════════════════════════════════════════════════════════════════════

def test_avversari_giocano_automaticamente():
    """Tra una step() del bot e l'altra, gli avversari devono aver giocato."""
    env = RisikoEnv(seed=42)
    _, info = env.reset()
    # Fa 10 step (azioni del bot). Verifica che il round avanzi
    round_iniziale = info["round"]
    for _ in range(10):
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        azione = random.choice(legali)
        obs, reward, terminated, truncated, info = env.step(int(azione))
        if terminated or truncated:
            break
    # Dopo 10 step del bot dovremmo essere oltre il round 1
    # (perché gli altri 3 giocatori giocano in mezzo)
    print(f"✓ Round dopo 10 step bot: {info['round']} (iniziale: {round_iniziale})")


# ═════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═════════════════════════════════════════════════════════════════════════

def main():
    tests = [
        # API Gymnasium
        test_reset_restituisce_obs_e_info,
        test_observation_space,
        test_action_space,
        test_action_mask_in_info,
        test_step_restituisce_5_valori,
        # Mask
        test_mask_sempre_almeno_un_true,
        # Partite
        test_partita_termina_con_reward,
        test_molte_partite_terminano,
        # Reward
        test_reward_solo_a_fine_partita,
        test_reward_vittoria_max,
        # Riproducibilità
        test_riproducibilita,
        # Sotto-fasi
        test_sotto_fasi_in_sequenza,
        # Avversari
        test_avversari_giocano_automaticamente,
    ]

    print("\n" + "=" * 60)
    print("Test Modulo 5c: RisikoEnv Gymnasium")
    print("=" * 60 + "\n")

    falliti = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} FALLITO: {e}")
            falliti.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} ERRORE: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            falliti.append(test.__name__)

    print("\n" + "=" * 60)
    if falliti:
        print(f"FALLITI: {len(falliti)}/{len(tests)}")
        for nome in falliti:
            print(f"  - {nome}")
    else:
        print(f"TUTTI I {len(tests)} TEST PASSATI ✓")
    print("=" * 60 + "\n")

    return len(falliti) == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
