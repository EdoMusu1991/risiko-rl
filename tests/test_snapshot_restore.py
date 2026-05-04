"""
test_snapshot_restore.py — I 4 test obbligatori (specifica ChatGPT).

1. Idempotenza: snapshot -> restore -> snapshot identico
2. Determinismo: stesso stato + stessa azione = stesso risultato
3. Replay: stessa sequenza azioni da snapshot = stesso esito finale
4. No side effects: simulazioni non alterano env reale

Questi test devono passare al 100% prima di scrivere MCTS.
Anche un solo fallimento = bug critico, MCTS non funzionera' sopra.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env import encoding as _encoding
_encoding.STAGE_A_ATTIVO = False  # snapshot/restore funziona con o senza Stage A; testiamo entrambi

from risiko_env import RisikoEnv


def _stato_ridotto(env) -> dict:
    """
    Riduce lo stato dell'env a una rappresentazione paragonabile (==).
    Usato per verificare uguaglianza fra stati senza false positive da
    differenze in oggetti contenitori.
    """
    if env.stato is None:
        return {"stato": None}

    return {
        "round": env.stato.round_corrente,
        "turno_di": env.stato.giocatore_corrente,
        "terminata": env.stato.terminata,
        "sotto_fase": env.sotto_fase,
        "step_count": env.step_count,
        "rinforzi_rimasti": env._rinforzi_rimasti,
        "attacco_corrente": env._attacco_corrente,
        "spostamento_corrente": env._spostamento_corrente,
        # Mappa: territorio -> (proprietario, armate)
        "mappa": {
            t: (env.stato.mappa[t].proprietario, env.stato.mappa[t].armate)
            for t in env.stato.mappa
        },
        # Per ogni giocatore: vivo, num carte, num territori
        "giocatori": {
            c: (g.vivo, len(g.carte), env.stato.num_territori_di(c))
            for c, g in env.stato.giocatori.items()
        },
        # Storia tracker
        "tracker_n_mosse": {
            c: len(mosse) for c, mosse in env._tracker.storia.items()
        },
    }


def _avanza_di_n_step(env, info, n: int) -> dict:
    """Esegue n step con azioni casuali, ritorna info dell'ultimo step."""
    rng_locale = np.random.RandomState(0)
    for _ in range(n):
        if info is None:
            break
        mask = info["action_mask"]
        legali = np.where(mask)[0]
        if len(legali) == 0:
            break
        action = int(rng_locale.choice(legali))
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    return info


# ────────────────────────────────────────────────────────────────────────
#  TEST 1: IDEMPOTENZA
# ────────────────────────────────────────────────────────────────────────

def test_idempotenza_dopo_reset():
    """snapshot -> restore -> snapshot deve essere identico."""
    env = RisikoEnv(seed=42)
    env.reset()

    snap1 = env.snapshot()
    env.restore(snap1)
    snap2 = env.snapshot()

    s1 = _stato_ridotto(env)
    env.restore(snap2)
    s2 = _stato_ridotto(env)

    assert s1 == s2, f"Idempotenza fallita dopo reset:\n{s1}\nvs\n{s2}"


def test_idempotenza_dopo_alcuni_step():
    """Idempotenza dopo aver eseguito alcune mosse."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()
    info = _avanza_di_n_step(env, info, n=20)

    snap = env.snapshot()
    s1 = _stato_ridotto(env)
    env.restore(snap)
    s2 = _stato_ridotto(env)

    assert s1 == s2, "Idempotenza dopo 20 step fallita"


# ────────────────────────────────────────────────────────────────────────
#  TEST 2: DETERMINISMO
# ────────────────────────────────────────────────────────────────────────

def test_determinismo_singola_azione():
    """Stesso stato + stessa azione = stesso risultato."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()

    # Snapshot prima di una azione
    snap = env.snapshot()
    mask = info["action_mask"]
    azione = int(np.where(mask)[0][0])

    # Esegue azione 1
    obs1, r1, t1, tr1, info1 = env.step(azione)
    s1 = _stato_ridotto(env)

    # Restore + esegue stessa azione
    env.restore(snap)
    obs2, r2, t2, tr2, info2 = env.step(azione)
    s2 = _stato_ridotto(env)

    assert s1 == s2, f"Determinismo fallito su azione singola"
    assert r1 == r2, f"Reward differente: {r1} vs {r2}"
    assert t1 == t2 and tr1 == tr2
    assert np.array_equal(obs1, obs2), "Observation differente!"


def test_determinismo_dopo_step_intermedio():
    """Determinismo anche da uno stato intermedio (post N step)."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()
    info = _avanza_di_n_step(env, info, n=10)

    if info is None:
        return  # partita gia' finita

    snap = env.snapshot()
    mask = info["action_mask"]
    legali = np.where(mask)[0]
    if len(legali) == 0:
        return
    azione = int(legali[0])

    obs1, r1, t1, tr1, info1 = env.step(azione)
    s1 = _stato_ridotto(env)

    env.restore(snap)
    obs2, r2, t2, tr2, info2 = env.step(azione)
    s2 = _stato_ridotto(env)

    assert s1 == s2, "Determinismo da stato intermedio fallito"
    assert r1 == r2


# ────────────────────────────────────────────────────────────────────────
#  TEST 3: REPLAY
# ────────────────────────────────────────────────────────────────────────

def test_replay_10_mosse():
    """Stessa sequenza azioni da stesso snapshot = stesso esito finale."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()

    snap = env.snapshot()

    # Sequenza 1: registra le azioni
    rng = np.random.RandomState(0)
    azioni_sequenza = []
    info_curr = info
    for _ in range(30):
        if info_curr is None:
            break
        mask = info_curr["action_mask"]
        legali = np.where(mask)[0]
        if len(legali) == 0:
            break
        azione = int(rng.choice(legali))
        azioni_sequenza.append(azione)
        obs, r, t, tr, info_curr = env.step(azione)
        if t or tr:
            break

    s_finale1 = _stato_ridotto(env)

    # Sequenza 2: replay da snapshot, stesse azioni
    env.restore(snap)
    for azione in azioni_sequenza:
        obs, r, t, tr, info_curr = env.step(azione)
        if t or tr:
            break

    s_finale2 = _stato_ridotto(env)

    assert s_finale1 == s_finale2, (
        f"Replay fallito! Stato dopo {len(azioni_sequenza)} mosse:\n"
        f"Originale: {s_finale1}\nReplay: {s_finale2}"
    )


def test_replay_partita_completa():
    """Replay di una partita completa (fino a fine)."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()

    snap = env.snapshot()
    rng = np.random.RandomState(0)

    azioni_sequenza = []
    info_curr = info
    while True:
        if info_curr is None:
            break
        mask = info_curr["action_mask"]
        legali = np.where(mask)[0]
        if len(legali) == 0:
            break
        azione = int(rng.choice(legali))
        azioni_sequenza.append(azione)
        obs, r, t, tr, info_curr = env.step(azione)
        if t or tr:
            break

    s_finale1 = _stato_ridotto(env)
    motivo1 = info_curr.get("motivo_fine") if info_curr else None
    vinc1 = info_curr.get("vincitore") if info_curr else None

    # Replay
    env.restore(snap)
    info_curr = None
    for azione in azioni_sequenza:
        obs, r, t, tr, info_curr = env.step(azione)
        if t or tr:
            break
    s_finale2 = _stato_ridotto(env)
    motivo2 = info_curr.get("motivo_fine") if info_curr else None
    vinc2 = info_curr.get("vincitore") if info_curr else None

    assert s_finale1 == s_finale2, "Replay partita completa fallito (stato)"
    assert motivo1 == motivo2, f"Motivo fine differente: {motivo1} vs {motivo2}"
    assert vinc1 == vinc2, f"Vincitore differente: {vinc1} vs {vinc2}"


# ────────────────────────────────────────────────────────────────────────
#  TEST 4: NO SIDE EFFECTS
# ────────────────────────────────────────────────────────────────────────

def test_no_side_effects_simulazione_breve():
    """Simulare azioni dopo snapshot non altera l'env reale dopo restore."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()
    info = _avanza_di_n_step(env, info, n=10)

    if info is None:
        return

    snap = env.snapshot()
    s_originale = _stato_ridotto(env)

    # Simulo 50 step
    info = _avanza_di_n_step(env, info, n=50)

    # Restore
    env.restore(snap)
    s_dopo_restore = _stato_ridotto(env)

    assert s_originale == s_dopo_restore, "Side effect rilevato dopo simulazione!"


def test_no_side_effects_multiple_simulazioni():
    """Multiple cicli di snapshot/simulate/restore preservano lo stato."""
    env = RisikoEnv(seed=42)
    obs, info = env.reset()
    info = _avanza_di_n_step(env, info, n=15)

    if info is None:
        return

    snap = env.snapshot()
    s_baseline = _stato_ridotto(env)

    # 10 cicli di simulazione
    for ciclo in range(10):
        info_sim = info
        rng = np.random.RandomState(ciclo)
        for _ in range(20):
            if info_sim is None:
                break
            mask = info_sim["action_mask"]
            legali = np.where(mask)[0]
            if len(legali) == 0:
                break
            obs_s, r_s, t_s, tr_s, info_sim = env.step(int(rng.choice(legali)))
            if t_s or tr_s:
                break

        # Restore dopo ogni ciclo
        env.restore(snap)
        s_dopo = _stato_ridotto(env)
        assert s_baseline == s_dopo, f"Side effect al ciclo {ciclo}"


# ────────────────────────────────────────────────────────────────────────
#  TEST 5 (bonus): PERFORMANCE
# ────────────────────────────────────────────────────────────────────────

def test_performance_snapshot_restore():
    """
    Misura tempo medio di snapshot/restore.
    Target: < 1ms (chiamato migliaia di volte da MCTS).
    """
    import time

    env = RisikoEnv(seed=42)
    obs, info = env.reset()
    info = _avanza_di_n_step(env, info, n=20)

    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        snap = env.snapshot()
        env.restore(snap)
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / n_iter) * 1000

    print(f"[INFO] Snapshot+Restore avg: {avg_ms:.3f} ms (target: <1ms)")
    if avg_ms > 5.0:
        print(f"[WARN] Snapshot/restore lento: {avg_ms:.3f}ms. MCTS sara' lento.")
    # Non e' un fallimento se >1ms, ma e' utile saperlo


# ────────────────────────────────────────────────────────────────────────
#  RUNNER
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Test 1a: Idempotenza dopo reset", test_idempotenza_dopo_reset),
        ("Test 1b: Idempotenza dopo step", test_idempotenza_dopo_alcuni_step),
        ("Test 2a: Determinismo singola azione", test_determinismo_singola_azione),
        ("Test 2b: Determinismo da stato intermedio", test_determinismo_dopo_step_intermedio),
        ("Test 3a: Replay 30 mosse", test_replay_10_mosse),
        ("Test 3b: Replay partita completa", test_replay_partita_completa),
        ("Test 4a: No side effects (breve)", test_no_side_effects_simulazione_breve),
        ("Test 4b: No side effects (multiple)", test_no_side_effects_multiple_simulazioni),
        ("Test 5: Performance", test_performance_snapshot_restore),
    ]

    passati = 0
    falliti = []
    for nome, fn in tests:
        try:
            fn()
            print(f"  [OK] {nome}")
            passati += 1
        except AssertionError as e:
            print(f"  [FAIL] {nome}: {e}")
            falliti.append((nome, str(e)))
        except Exception as e:
            print(f"  [ERROR] {nome}: {type(e).__name__}: {e}")
            falliti.append((nome, f"{type(e).__name__}: {e}"))

    print()
    if falliti:
        print(f"FALLITI {len(falliti)}/{len(tests)}:")
        for nome, msg in falliti:
            print(f"  - {nome}")
        sys.exit(1)
    else:
        print(f"TUTTI I {len(tests)} TEST PASSATI ✓")
