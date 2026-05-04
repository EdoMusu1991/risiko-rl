"""
test_alphazero_pipeline_mini.py — Pipeline sequenziale: self-play -> buffer -> train.

Settimana 5, sub-step 4 (validato da ChatGPT: prima correttezza, poi velocita').

Idea:
- Poche partite self-play (5) con n_sim basso (per non superare timeout)
- Tutti i sample -> ReplayBuffer
- Mini-training (200 step) con dati REALI di MCTS
- Verifica che la loss scenda

Non e' un test "il bot e' forte". E' un test "i dati reali di MCTS sono
apprendibili dalla rete". Se questo passa, la pipeline e' coerente
end-to-end e si puo' passare al training reale (su GPU).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch

from risiko_env import encoding as e
e.STAGE_A_ATTIVO = True
from risiko_env import RisikoEnv

from alphazero.network import RisikoNet
from alphazero.selfplay import gioca_partita_selfplay
from alphazero.training import ReplayBuffer, Trainer


def test_pipeline_mini_sequenziale():
    """
    Pipeline mini end-to-end:
    1. 5 partite self-play -> buffer
    2. 200 step training
    3. Verifica loss scende
    """
    print("\n" + "=" * 60)
    print("PIPELINE MINI: self-play -> buffer -> train (dati REALI)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # ─── 1. Genera partite self-play ────────────────────────
    net = RisikoNet()
    buffer = ReplayBuffer(max_size=5000, seed=42)
    
    n_partite = 5
    n_sim_per_partita = 5
    max_decisioni = 60
    
    print(f"\nGenerazione {n_partite} partite self-play")
    print(f"  n_sim={n_sim_per_partita}, max_decisioni={max_decisioni}")
    
    t0 = time.perf_counter()
    n_samples_totali = 0
    
    for i in range(n_partite):
        env = RisikoEnv(seed=100 + i, mode_1v1=True, reward_mode='margin')
        samples, stats = gioca_partita_selfplay(
            env, net,
            n_simulations=n_sim_per_partita,
            seed=100 + i,
            max_decisioni=max_decisioni,
        )
        buffer.add_partita(samples)
        n_samples_totali += len(samples)
        print(f"  Partita {i+1}: {len(samples)} sample, "
              f"reward={stats['reward_finale']:+.3f}, "
              f"vincitore={stats.get('vincitore')}")
    
    t_partite = time.perf_counter() - t0
    print(f"\nTempo totale generazione: {t_partite:.1f}s ({t_partite/n_partite:.1f}s/partita)")
    print(f"Buffer popolato: {len(buffer)} sample")
    
    assert len(buffer) >= 50, f"Buffer troppo piccolo: {len(buffer)}"
    
    # ─── 2. Training mini ───────────────────────────────────
    trainer = Trainer(net, lr=0.001, weight_decay=1e-4)
    
    n_train_steps = 200
    batch_size = 32
    
    print(f"\nTraining {n_train_steps} step su batch={batch_size}, lr=0.001")
    print(f"{'Step':>5} | {'Total':>7} | {'Value':>7} | {'Policy':>7} | {'GradN':>6}")
    print("-" * 50)
    
    losses = []
    value_losses = []
    policy_losses = []
    
    t0 = time.perf_counter()
    for step in range(n_train_steps):
        batch_samples = buffer.sample(batch_size=batch_size)
        m = trainer.train_step(batch_samples)
        
        losses.append(m["total_loss"])
        value_losses.append(m["value_loss"])
        policy_losses.append(m["policy_loss"])
        
        if step % 25 == 0 or step == n_train_steps - 1:
            print(f"{step:>5} | {m['total_loss']:>7.4f} | "
                  f"{m['value_loss']:>7.4f} | {m['policy_loss']:>7.4f} | "
                  f"{m['grad_norm']:>6.2f}")
    
    t_train = time.perf_counter() - t0
    print(f"\nTempo training: {t_train:.1f}s ({t_train/n_train_steps*1000:.1f}ms/step)")
    
    # ─── 3. Verifica che la loss scenda ─────────────────────
    loss_inizio = sum(losses[:10]) / 10
    loss_fine = sum(losses[-10:]) / 10
    riduzione = (loss_inizio - loss_fine) / loss_inizio * 100
    
    v_inizio = sum(value_losses[:10]) / 10
    v_fine = sum(value_losses[-10:]) / 10
    
    p_inizio = sum(policy_losses[:10]) / 10
    p_fine = sum(policy_losses[-10:]) / 10
    
    print(f"\n--- Risultati ---")
    print(f"Loss totale:  {loss_inizio:.4f} -> {loss_fine:.4f}  "
          f"({riduzione:+.1f}%)")
    print(f"Value loss:   {v_inizio:.4f} -> {v_fine:.4f}")
    print(f"Policy loss:  {p_inizio:.4f} -> {p_fine:.4f}")
    
    # ─── ASSERTIONS ─────────────────────────────────────────
    # Su dati reali la riduzione sara' piu' modesta del test sintetico
    # (~30-60% e' ragionevole per 200 step su pochi dati).
    # NB: Dati MCTS hanno target piu' "rumorosi" del nostro pattern lineare.
    
    assert riduzione > 15, \
        f"Loss totale non scende abbastanza: {riduzione:.1f}% (atteso >15%)"
    
    assert all(np.isfinite(l) for l in losses), \
        "NaN o Inf nella loss!"
    
    print(f"\n✅ PIPELINE MINI FUNZIONA")
    print(f"   Dati reali di MCTS apprendibili dalla rete.")
    print(f"   Si puo' procedere al training reale (su GPU).")


def test_pipeline_consistenza_state():
    """
    Verifica che il flow self-play -> buffer -> train non corrompa
    nessuno stato (env, samples, network).
    """
    print("\n" + "=" * 60)
    print("TEST CONSISTENZA STATE (samples + buffer + trainer)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Genera 1 partita
    env = RisikoEnv(seed=42, mode_1v1=True, reward_mode='margin')
    net = RisikoNet()
    samples, _ = gioca_partita_selfplay(
        env, net,
        n_simulations=3,
        seed=42,
        max_decisioni=20,
    )
    
    # Verifica ogni sample
    for i, s in enumerate(samples):
        # obs valido
        assert s.obs.shape == (342,)
        assert 0.0 <= s.obs.min() and s.obs.max() <= 1.0
        # mask valida
        assert s.mask.sum() >= 2
        # policy somma a 1, zero su illegali
        assert abs(s.policy_target.sum() - 1.0) < 1e-4
        assert s.policy_target[~s.mask].sum() < 1e-6
        # value in range
        assert -1.0 <= s.value_target <= 1.0
        # player valido
        assert s.player_at_state in ("BLU", "ROSSO")
    
    # Aggiungi al buffer
    buffer = ReplayBuffer(max_size=100, seed=42)
    buffer.add_partita(samples)
    assert len(buffer) == len(samples)
    
    # Sample dal buffer e verifica che siano gli stessi tipi
    batch = buffer.sample(batch_size=4)
    for s in batch:
        assert hasattr(s, "obs")
        assert hasattr(s, "policy_target")
        assert hasattr(s, "value_target")
    
    # Training step
    trainer = Trainer(net, lr=0.001)
    metrics = trainer.train_step(batch)
    assert all(np.isfinite(v) for k, v in metrics.items() if k != "step")
    
    print(f"  ✓ Pipeline produce dati consistenti ({len(samples)} sample, training ok)")


if __name__ == "__main__":
    print("Test Pipeline MINI Settimana 5 (sub-step 4)")
    
    test_pipeline_consistenza_state()
    test_pipeline_mini_sequenziale()
    
    print("\n" + "=" * 60)
    print("TUTTI I 2 TEST PASSATI ✓")
    print("Pipeline self-play -> buffer -> train VALIDATA.")
