# Diario Settimana 5 — Replay buffer, Trainer, Pipeline mini

**Stato**: Sub-step 1-4 completati ✅ (eval pipeline rinviata)
**Data**: maggio 2026

## Obiettivi della settimana

> Settimana 5 (validato da ChatGPT): replay buffer + training loop + test overfit + pipeline mini.
> Regola: prima correttezza, poi velocita', poi scala.

## Sub-step completati

### Sub-step 1 — ReplayBuffer ✅
File: `alphazero/training/replay_buffer.py`

- `ReplayBuffer(max_size=100k)`: deque FIFO, sampling uniforme con replacement
- `samples_to_batch()`: converte TrainingSample in dict di tensori PyTorch
- 12 test verdi (creazione, FIFO, sample, conversione, compatibilita' rete)

### Sub-step 2 — Trainer ✅
File: `alphazero/training/trainer.py`

- `Trainer(net, lr=0.001, weight_decay=1e-4)`: Adam + grad clip a 1.0
- `train_step(samples) -> metrics`
- `save_checkpoint() / load_checkpoint()`
- Codice agnostico CPU/GPU (parametro `device`)

### Sub-step 3 — Test overfitting (CRITICO) ✅
File: `tests/test_alphazero_overfit.py`

100 sample sintetici fissi con pattern imparabile, 300 step di training:
- **Loss totale: 2.29 → 0.05 (-97.7%)**
- Value loss: 0.054 → 0.009 (-84%)
- Policy loss: 2.24 → 0.04 (-98%)
- Gradienti sani, no NaN

ChatGPT aveva detto: *"Se la loss non scende, training loop e' rotto. Non andare avanti."*
**Loss scende drasticamente. Training loop FUNZIONANTE.**

### Sub-step 4 — Pipeline mini sequenziale ✅
File: `tests/test_alphazero_pipeline_mini.py`

5 partite self-play (n_sim=5, max_decisioni=60) → buffer (300 sample) → 200 step training:
- **Tempo generazione partite: 6.7s/partita su CPU**
- **Tempo training: 6.9ms/step**
- Loss totale: 2.36 → 1.13 (-52%)
- Policy loss scende dimostrando apprendimento da dati MCTS reali

#### Limite riconosciuto (onesto)
Tutte le partite di test finiscono con `reward=0` (troncate a 60 decisioni). Quindi:
- Value loss resta bassissima dall'inizio (~0.0005-0.0015)
- Il value learning NON e' ancora validato su dati reali

Questo limite verra' superato in Settimana 6 con partite complete su GPU.

## Sub-step rinviato

### Sub-step 5 — Eval pipeline
gen N vs random, gen N vs gen N-1: rinviato a Settimana 6 quando avremo
checkpoint reali da confrontare.

## Statistiche finali

- **237 test verdi totali**
- +17 test nuovi (12 buffer + 3 overfit + 2 pipeline)
- Codice nuovo: ~600 righe in `alphazero/training/`

## Performance reali (CPU)

- Train step (batch=32): ~7ms
- Partita self-play (n_sim=5, max=60 decisioni): ~7s
- Su GPU atteso speedup 5-10x sia per partite che training

## Decisioni di design prese in Settimana 5

1. **lr=0.001** di partenza, weight_decay=1e-4 (ChatGPT)
2. **Gradient clipping** a max_norm=1.0 (evita esplosioni in early training)
3. **Sampling con replacement** dal buffer (standard AlphaZero)
4. **Adam** come optimizer (no SGD)
5. **Codice agnostico CPU/GPU** (parametro device)
6. **Pipeline sequenziale** prima di parallelismo (regola di ChatGPT)

## Cosa NON ho ancora fatto

- ❌ Eval pipeline (Sub-step 5) - rinviata
- ❌ Test su GPU - servira' Colab Pro+
- ❌ Partite reali complete (solo troncate per timing CPU)
- ❌ Warm-start con rollout euristico (opzionale)
- ❌ Dirichlet noise (ottimizzazione)

## Settimana 6 plan

1. Setup Colab Pro+ con GPU
2. Genera 50-100 partite COMPLETE (n_sim=50-100)
3. Training su buffer reale per 1000-5000 step
4. Salva checkpoint
5. Eval: gen 1 vs random (target: >50% win rate)
6. Confronto preliminare vs baseline 29% PPO

## Lezioni apprese

1. **Test overfit funziona perfettamente come "smoke test" del training loop**.
   Senza questo, avrei rischiato di andare avanti con un training loop rotto
   senza accorgermene.

2. **Su CPU partite Risiko sono troppo lente per training reale**.
   GPU diventa indispensabile a partire dalla Settimana 6.

3. **Reward sparso e' un problema**: con max_decisioni=60, le partite finiscono
   troncate (reward=0) e il value head non ha segnale. Solo partite intere
   producono value reali da imparare.

4. **Time/step training e' veloce** (~7ms su CPU). Il bottleneck e' la
   generazione delle partite, non il training.
