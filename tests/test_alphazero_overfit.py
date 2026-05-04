"""
test_alphazero_overfit.py — TEST CRITICO di overfitting.

Settimana 5, sub-step 3.

Idea (validata da ChatGPT):
- Prendi 100 sample fissi
- Trainare la rete SOLO su quei 100 sample
- Se la loss NON SCENDE significativamente -> training loop e' rotto
- Se la loss scende -> training loop funziona, possiamo passare al training reale

QUESTO TEST E' L'UNICO MODO PER ESSERE SICURI CHE IL TRAINING LOOP NON SIA
SILENZIOSAMENTE ROTTO. Senza, scopri il bug solo dopo 1000 partite di
self-play e settimane di tempo perso.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from alphazero.network import RisikoNet, ACTION_DIM
from alphazero.training import Trainer, ReplayBuffer
from alphazero.selfplay.self_play import TrainingSample


def make_sintetico_sample(rng, n_legali=10):
    """
    Crea un TrainingSample sintetico con un PATTERN imparabile:
    - obs random in [0,1]
    - mask con n_legali azioni
    - policy_target one-hot sulla prima azione legale (pattern facile)
    - value_target = funzione lineare delle prime 5 feature dell'obs
    """
    obs = rng.random(342, dtype=np.float64).astype(np.float32)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    legal_idxs = rng.choice(ACTION_DIM, size=n_legali, replace=False)
    mask[legal_idxs] = True
    
    # Policy: one-hot su prima azione legale
    policy_target = np.zeros(ACTION_DIM, dtype=np.float32)
    policy_target[legal_idxs[0]] = 1.0
    
    # Value: pattern lineare delle prime 5 feature
    value = float(np.clip(obs[:5].sum() / 2.5 - 1.0, -1.0, 1.0))
    
    return TrainingSample(
        obs=obs,
        mask=mask,
        policy_target=policy_target,
        player_at_state="BLU",
        value_target=value,
    )


def test_overfit_su_100_sample():
    """
    TEST CRITICO: la rete deve overfittare 100 sample fissi.
    
    Se questo test fallisce, il training loop e' rotto e NON andiamo
    avanti finche' non e' aggiustato.
    """
    print("\n" + "=" * 60)
    print("TEST OVERFITTING - 100 sample fissi")
    print("=" * 60)
    
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    
    # ─── 1. Genera dataset fisso di 100 sample ──────────────
    samples = [make_sintetico_sample(rng) for _ in range(100)]
    print(f"Dataset: 100 sample sintetici con pattern imparabile")
    
    # ─── 2. Carica nel buffer ───────────────────────────────
    buffer = ReplayBuffer(max_size=200, seed=42)
    buffer.add_partita(samples)
    assert len(buffer) == 100
    
    # ─── 3. Crea trainer e rete ─────────────────────────────
    net = RisikoNet()
    trainer = Trainer(net, lr=0.001, weight_decay=1e-4)
    
    # ─── 4. Train 300 step su batch di 32 ───────────────────
    n_steps = 300
    batch_size = 32
    
    losses = []
    value_losses = []
    policy_losses = []
    
    print(f"\nTraining {n_steps} step su batch={batch_size}, lr=0.001")
    print(f"{'Step':>5} | {'Total':>7} | {'Value':>7} | {'Policy':>7} | {'GradN':>6}")
    print("-" * 50)
    
    for step in range(n_steps):
        batch_samples = buffer.sample(batch_size=batch_size)
        metrics = trainer.train_step(batch_samples)
        
        losses.append(metrics["total_loss"])
        value_losses.append(metrics["value_loss"])
        policy_losses.append(metrics["policy_loss"])
        
        if step % 50 == 0 or step == n_steps - 1:
            print(f"{step:>5} | {metrics['total_loss']:>7.4f} | "
                  f"{metrics['value_loss']:>7.4f} | {metrics['policy_loss']:>7.4f} | "
                  f"{metrics['grad_norm']:>6.2f}")
    
    # ─── 5. Verifica che la loss sia scesa ──────────────────
    loss_inizio = sum(losses[:10]) / 10  # media primi 10
    loss_fine = sum(losses[-10:]) / 10   # media ultimi 10
    riduzione_perc = (loss_inizio - loss_fine) / loss_inizio * 100
    
    v_inizio = sum(value_losses[:10]) / 10
    v_fine = sum(value_losses[-10:]) / 10
    
    p_inizio = sum(policy_losses[:10]) / 10
    p_fine = sum(policy_losses[-10:]) / 10
    
    print(f"\n--- Risultati ---")
    print(f"Loss totale:  {loss_inizio:.4f} -> {loss_fine:.4f}  "
          f"({riduzione_perc:+.1f}%)")
    print(f"Value loss:   {v_inizio:.4f} -> {v_fine:.4f}")
    print(f"Policy loss:  {p_inizio:.4f} -> {p_fine:.4f}")
    
    # ─── ASSERTIONS ───────────────────────────────────────
    # 1. Loss deve scendere significativamente
    assert riduzione_perc > 30, \
        f"Loss totale non scende abbastanza: {riduzione_perc:.1f}% (atteso >30%)"
    
    # 2. Value loss deve scendere
    assert v_fine < v_inizio * 0.5, \
        f"Value loss non scende abbastanza: {v_inizio:.4f} -> {v_fine:.4f}"
    
    # 3. Policy loss deve scendere (overfit one-hot e' relativamente facile)
    assert p_fine < p_inizio * 0.7, \
        f"Policy loss non scende abbastanza: {p_inizio:.4f} -> {p_fine:.4f}"
    
    # 4. Loss deve essere finita (no NaN/Inf)
    assert all(np.isfinite(l) for l in losses), "NaN o Inf nella loss!"
    
    print(f"\n✅ TEST OVERFITTING PASSATO")
    print(f"   La rete IMPARA dai dati. Training loop funzionante.")
    print(f"   Possiamo procedere al training reale.")


def test_overfit_value_solo():
    """
    Test piu' mirato: solo value loss su pattern semplice.
    Se questo non funziona, il problema e' nella rete o nel value head.
    """
    print("\n" + "=" * 60)
    print("TEST VALUE-ONLY (pattern lineare)")
    print("=" * 60)
    
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    
    samples = [make_sintetico_sample(rng) for _ in range(200)]
    buffer = ReplayBuffer(max_size=200, seed=42)
    buffer.add_partita(samples)
    
    net = RisikoNet()
    trainer = Trainer(net, lr=0.001, weight_decay=1e-4)
    
    v_losses = []
    for step in range(200):
        batch = buffer.sample(batch_size=32)
        m = trainer.train_step(batch)
        v_losses.append(m["value_loss"])
    
    v_start = sum(v_losses[:10]) / 10
    v_end = sum(v_losses[-10:]) / 10
    print(f"Value loss: {v_start:.4f} -> {v_end:.4f}  "
          f"({(v_start - v_end) / v_start * 100:+.1f}%)")
    
    assert v_end < v_start * 0.3, \
        f"Value loss non scende abbastanza: {v_start} -> {v_end}"
    print(f"  ✓ Value head impara il pattern lineare")


def test_checkpoint_save_load():
    """Verifica che salvare/caricare un checkpoint funzioni."""
    import tempfile
    
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    samples = [make_sintetico_sample(rng) for _ in range(50)]
    
    net1 = RisikoNet()
    trainer1 = Trainer(net1, lr=0.001)
    
    # Train qualche step
    buffer = ReplayBuffer(max_size=100, seed=42)
    buffer.add_partita(samples)
    for _ in range(10):
        batch = buffer.sample(batch_size=8)
        trainer1.train_step(batch)
    
    step_dopo_training = trainer1.step_counter
    
    # Salva
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    trainer1.save_checkpoint(ckpt_path)
    
    # Crea nuova rete + trainer, carica
    net2 = RisikoNet()
    trainer2 = Trainer(net2, lr=0.001)
    trainer2.load_checkpoint(ckpt_path)
    
    # Verifica che i pesi siano uguali
    sd1 = net1.state_dict()
    sd2 = net2.state_dict()
    for key in sd1:
        assert torch.allclose(sd1[key], sd2[key]), f"Mismatch key {key}"
    
    # Verifica step counter
    assert trainer2.step_counter == step_dopo_training
    
    # Cleanup
    os.unlink(ckpt_path)
    print(f"  ✓ Checkpoint save/load funziona (step={step_dopo_training})")


if __name__ == "__main__":
    print("Test Trainer + Overfit (alphazero/training/)")
    print("=" * 60)
    
    test_overfit_su_100_sample()
    test_overfit_value_solo()
    test_checkpoint_save_load()
    
    print("\n" + "=" * 60)
    print("TUTTI I 3 TEST PASSATI ✓")
