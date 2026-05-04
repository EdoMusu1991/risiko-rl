"""
test_alphazero_replay_buffer.py — Test del replay buffer.

Settimana 5, sub-step 1. Verifica:
- Aggiunta sample (singoli e in batch da partita)
- Sampling uniforme
- Maxlen rispettato (FIFO)
- Conversione TrainingSample -> tensori PyTorch corretta
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from alphazero.training.replay_buffer import ReplayBuffer, samples_to_batch
from alphazero.selfplay.self_play import TrainingSample
from alphazero.network import ACTION_DIM


def make_dummy_sample(seed=0, player="BLU", value=0.5):
    """Helper: crea un TrainingSample fittizio per test."""
    rng = np.random.default_rng(seed)
    obs = rng.random(342, dtype=np.float64).astype(np.float32)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[:20] = True  # 20 azioni legali
    policy_target = np.zeros(ACTION_DIM, dtype=np.float32)
    policy_target[:20] = 1.0 / 20  # uniforme sulle legali
    return TrainingSample(
        obs=obs,
        mask=mask,
        policy_target=policy_target,
        player_at_state=player,
        value_target=value,
    )


# ─────────────────────────────────────────────────────────────────
#  TEST BUFFER
# ─────────────────────────────────────────────────────────────────

def test_buffer_creazione():
    buf = ReplayBuffer(max_size=1000)
    assert len(buf) == 0
    assert not buf.is_ready(min_size=1)
    print("  ✓ Buffer creato vuoto")


def test_buffer_add_singolo():
    buf = ReplayBuffer(max_size=100)
    s = make_dummy_sample()
    buf.add(s)
    assert len(buf) == 1
    print("  ✓ add() singolo funziona")


def test_buffer_add_partita():
    buf = ReplayBuffer(max_size=100)
    samples = [make_dummy_sample(seed=i) for i in range(15)]
    buf.add_partita(samples)
    assert len(buf) == 15
    print("  ✓ add_partita() aggiunge tutti")


def test_buffer_maxlen_fifo():
    """Quando il buffer e' pieno, scarta i piu' vecchi."""
    buf = ReplayBuffer(max_size=10)
    # Aggiungo 15 sample con value crescente
    for i in range(15):
        buf.add(make_dummy_sample(seed=i, value=float(i)))
    
    assert len(buf) == 10
    # I primi 5 devono essere stati scartati
    # I sample residui devono avere value 5..14
    values = sorted([s.value_target for s in buf.buffer])
    assert values == [float(i) for i in range(5, 15)], f"Values: {values}"
    print("  ✓ Maxlen rispettato (FIFO)")


def test_buffer_sample():
    buf = ReplayBuffer(max_size=100, seed=42)
    for i in range(50):
        buf.add(make_dummy_sample(seed=i))
    
    batch = buf.sample(batch_size=8)
    assert len(batch) == 8
    assert all(isinstance(s, TrainingSample) for s in batch)
    print("  ✓ sample(batch_size=8) restituisce 8 TrainingSample")


def test_buffer_sample_with_replacement():
    """Sampling con replacement: un sample puo' apparire piu' volte."""
    buf = ReplayBuffer(max_size=100, seed=42)
    # Solo 5 sample, batch 100 -> molti duplicati
    for i in range(5):
        buf.add(make_dummy_sample(seed=i, value=float(i)))
    
    batch = buf.sample(batch_size=100)
    assert len(batch) == 100
    # Distribuzione approssimativamente uniforme (~20 per ognuno)
    counts = {}
    for s in batch:
        v = s.value_target
        counts[v] = counts.get(v, 0) + 1
    assert len(counts) == 5  # tutti e 5 i sample sono stati pescati almeno una volta
    print(f"  ✓ Sampling with replacement: distribuzione {sorted(counts.values())}")


def test_buffer_is_ready():
    buf = ReplayBuffer(max_size=100)
    assert not buf.is_ready(min_size=10)
    
    for i in range(10):
        buf.add(make_dummy_sample(seed=i))
    assert buf.is_ready(min_size=10)
    assert not buf.is_ready(min_size=20)
    print("  ✓ is_ready() funziona")


def test_buffer_sample_vuoto_solleva_errore():
    """sample() su buffer vuoto deve sollevare ValueError."""
    buf = ReplayBuffer(max_size=10)
    try:
        buf.sample(batch_size=4)
        assert False, "Doveva sollevare ValueError"
    except ValueError:
        pass
    print("  ✓ sample() su buffer vuoto solleva ValueError")


# ─────────────────────────────────────────────────────────────────
#  TEST CONVERSIONE samples_to_batch
# ─────────────────────────────────────────────────────────────────

def test_samples_to_batch_shapes():
    """Verifica shape dei tensori."""
    samples = [make_dummy_sample(seed=i) for i in range(8)]
    batch = samples_to_batch(samples, device="cpu")
    
    assert batch["obs"].shape == (8, 342)
    assert batch["mask"].shape == (8, ACTION_DIM)
    assert batch["policy_target"].shape == (8, ACTION_DIM)
    assert batch["value_target"].shape == (8, 1)
    print("  ✓ samples_to_batch: shapes corrette")


def test_samples_to_batch_dtypes():
    """Verifica dtype dei tensori (importante per training)."""
    samples = [make_dummy_sample(seed=i) for i in range(4)]
    batch = samples_to_batch(samples, device="cpu")
    
    assert batch["obs"].dtype == torch.float32
    assert batch["mask"].dtype == torch.bool
    assert batch["policy_target"].dtype == torch.float32
    assert batch["value_target"].dtype == torch.float32
    print("  ✓ samples_to_batch: dtypes corretti")


def test_samples_to_batch_valori():
    """I valori devono corrispondere ai sample originali."""
    s1 = make_dummy_sample(seed=1, value=0.7)
    s2 = make_dummy_sample(seed=2, value=-0.3)
    batch = samples_to_batch([s1, s2], device="cpu")
    
    # Value target (con tolleranza float32)
    assert abs(batch["value_target"][0].item() - 0.7) < 1e-6
    assert abs(batch["value_target"][1].item() - (-0.3)) < 1e-6
    
    # Obs preserva i valori
    assert torch.allclose(batch["obs"][0], torch.from_numpy(s1.obs))
    
    # Mask preserva i bool
    assert batch["mask"][0].sum().item() == 20  # 20 azioni legali
    
    # Policy target somma a 1
    assert abs(batch["policy_target"][0].sum().item() - 1.0) < 1e-5
    print("  ✓ samples_to_batch: valori preservati")


def test_samples_to_batch_compatibile_con_rete():
    """Il batch deve essere compatibile con forward + loss della RisikoNet."""
    from alphazero.network import RisikoNet, alphazero_loss
    
    samples = [make_dummy_sample(seed=i) for i in range(4)]
    batch = samples_to_batch(samples, device="cpu")
    
    net = RisikoNet()
    policy_logits, value = net(batch["obs"])
    
    # Loss
    loss_dict = alphazero_loss(
        policy_logits, value,
        batch["policy_target"], batch["value_target"],
        batch["mask"],
    )
    
    assert torch.isfinite(loss_dict["total"])
    print(f"  ✓ Batch compatibile con rete (loss={loss_dict['total'].item():.3f})")


# ─────────────────────────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test ReplayBuffer + samples_to_batch")
    print("=" * 50)
    
    print("\nReplayBuffer:")
    test_buffer_creazione()
    test_buffer_add_singolo()
    test_buffer_add_partita()
    test_buffer_maxlen_fifo()
    test_buffer_sample()
    test_buffer_sample_with_replacement()
    test_buffer_is_ready()
    test_buffer_sample_vuoto_solleva_errore()
    
    print("\nsamples_to_batch:")
    test_samples_to_batch_shapes()
    test_samples_to_batch_dtypes()
    test_samples_to_batch_valori()
    test_samples_to_batch_compatibile_con_rete()
    
    print("=" * 50)
    print("TUTTI I 12 TEST PASSATI ✓")
