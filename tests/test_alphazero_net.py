"""
test_alphazero_net.py — Unit test per RisikoNet.

Test:
1. Creazione rete e conteggio parametri (~600k)
2. Forward single sample shape
3. Forward batch shape
4. Mask + softmax: prob illegali = 0, legali sommano a 1
5. Loss AlphaZero produce numero finito
6. Backward: gradient flow su tutti i parametri
7. Rete reagisce a input diversi (non e' costante)
8. Integrazione con observation reale di RisikoEnv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from alphazero.network import (
    RisikoNet,
    INPUT_DIM,
    ACTION_DIM,
    apply_mask_and_softmax,
    alphazero_loss,
)


def test_creazione_rete():
    net = RisikoNet()
    n_params = net.num_parameters()
    # Atteso ~600-700k parametri (validato design ChatGPT)
    assert 500_000 <= n_params <= 800_000, f"Parametri: {n_params}"
    print(f"  ✓ Rete creata, {n_params:,} parametri")


def test_forward_single_sample():
    net = RisikoNet()
    x = torch.rand(INPUT_DIM)
    policy_logits, value = net(x)
    assert policy_logits.shape == (1, ACTION_DIM)
    assert value.shape == (1, 1)
    assert -1.0 <= value.item() <= 1.0
    print(f"  ✓ Forward single sample: shapes corrette")


def test_forward_batch():
    net = RisikoNet()
    batch = torch.rand(32, INPUT_DIM)
    policy_b, value_b = net(batch)
    assert policy_b.shape == (32, ACTION_DIM)
    assert value_b.shape == (32, 1)
    print(f"  ✓ Forward batch (32): shapes corrette")


def test_mask_softmax():
    net = RisikoNet()
    x = torch.rand(INPUT_DIM)
    policy_logits, _ = net(x)

    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    mask[0:21] = True  # solo prime 21 azioni legali

    policy_dist = apply_mask_and_softmax(policy_logits[0], mask)

    assert abs(policy_dist[:21].sum().item() - 1.0) < 1e-5
    assert policy_dist[21:].sum().item() < 1e-5
    print(f"  ✓ Mask+softmax: prob legali=1, illegali=0")


def test_loss_alphazero():
    net = RisikoNet()
    batch = torch.rand(32, INPUT_DIM)
    policy_pred, value_pred = net(batch)

    v_target = torch.rand(32, 1) * 2 - 1
    p_target = torch.zeros(32, ACTION_DIM)
    mask_b = torch.zeros(32, ACTION_DIM, dtype=torch.bool)
    for i in range(32):
        mask_b[i, :21] = True
        p_target[i, :21] = torch.softmax(torch.rand(21), dim=0)

    loss_dict = alphazero_loss(policy_pred, value_pred, p_target, v_target, mask_b)
    assert torch.isfinite(loss_dict['total'])
    assert loss_dict['total'].item() > 0
    print(f"  ✓ Loss totale finita: {loss_dict['total'].item():.4f}")


def test_gradient_flow():
    net = RisikoNet()
    batch = torch.rand(8, INPUT_DIM)
    p_pred, v_pred = net(batch)

    # Target non-triviali: value diversi e policy distribuita
    v_target = torch.rand(8, 1) * 2 - 1
    p_target = torch.zeros(8, ACTION_DIM)
    mask_b = torch.zeros(8, ACTION_DIM, dtype=torch.bool)
    for i in range(8):
        mask_b[i, :20] = True
        p_target[i, :20] = torch.softmax(torch.rand(20), dim=0)

    ld = alphazero_loss(p_pred, v_pred, p_target, v_target, mask_b)
    ld['total'].backward()

    n_grad = sum(1 for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in net.parameters() if p.requires_grad)
    assert n_grad == n_total, f"Solo {n_grad}/{n_total} parametri hanno gradiente"
    print(f"  ✓ Gradient flow: {n_grad}/{n_total} params")


def test_rete_reagisce():
    net = RisikoNet()
    x1 = torch.zeros(INPUT_DIM)
    x2 = torch.ones(INPUT_DIM)
    p1, v1 = net(x1)
    p2, v2 = net(x2)
    diff = (p1 - p2).abs().mean().item()
    assert diff > 1e-4, "La rete non distingue zeros da ones"
    print(f"  ✓ Rete distingue input diversi (diff={diff:.4f})")


def test_integrazione_env():
    """La rete accetta observation reale dall'env."""
    from risiko_env import encoding as e
    e.STAGE_A_ATTIVO = True
    from risiko_env import RisikoEnv

    env = RisikoEnv(seed=42, mode_1v1=True, reward_mode='margin')
    obs, info = env.reset()
    assert obs.shape == (INPUT_DIM,), f"INPUT_DIM mismatch: {obs.shape} vs {INPUT_DIM}"

    net = RisikoNet()
    obs_t = torch.from_numpy(obs).float()
    with torch.no_grad():
        p_logits, value = net(obs_t)

    mask_t = torch.from_numpy(info['action_mask']).bool()
    policy = apply_mask_and_softmax(p_logits[0], mask_t)

    # Controlla che la policy abbia massa sulle azioni legali
    n_legali = mask_t.sum().item()
    assert abs(policy.sum().item() - 1.0) < 1e-5
    assert (policy[mask_t].sum().item() - 1.0) < 1e-5
    print(f"  ✓ Integrazione env OK ({n_legali} azioni legali iniziali)")


def test_rete_impara():
    """Test critico: la rete deve imparare un pattern semplice."""
    torch.manual_seed(42)
    net = RisikoNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    # Pattern: value = funzione lineare delle prime 10 feature
    N = 512
    X = torch.rand(N, INPUT_DIM)
    v_tgt = (X[:, :10].sum(dim=1) / 5.0 - 1.0).clamp(-1, 1).unsqueeze(1)
    p_tgt = torch.zeros(N, ACTION_DIM)
    p_tgt[:, 0] = 1.0
    mask_t = torch.zeros(N, ACTION_DIM, dtype=torch.bool)
    mask_t[:, :5] = True

    v_losses = []
    for step in range(150):
        idx = torch.randint(0, N, (32,))
        p_pred, v_pred = net(X[idx])
        ld = alphazero_loss(p_pred, v_pred, p_tgt[idx], v_tgt[idx], mask_t[idx])
        optimizer.zero_grad()
        ld['total'].backward()
        optimizer.step()
        v_losses.append(ld['value_loss'].item())

    v_inizio = sum(v_losses[:10]) / 10
    v_fine = sum(v_losses[-10:]) / 10
    riduzione = v_inizio - v_fine
    assert riduzione > 0.005, f"V_loss non scende: {v_inizio:.4f} -> {v_fine:.4f}"
    print(f"  ✓ Rete impara pattern (v_loss {v_inizio:.4f} -> {v_fine:.4f})")


if __name__ == "__main__":
    print("Test RisikoNet (alphazero/network/model.py)")
    print("=" * 55)
    test_creazione_rete()
    test_forward_single_sample()
    test_forward_batch()
    test_mask_softmax()
    test_loss_alphazero()
    test_gradient_flow()
    test_rete_reagisce()
    test_integrazione_env()
    test_rete_impara()
    print("=" * 55)
    print("TUTTI I 9 TEST PASSATI ✓")
