# risiko-rl

Simulatore RisiKo in Python (variante torneo italiano) come ambiente Gymnasium per addestramento di un bot RL.

## Stato del progetto

- **Modulo 1** (dati statici e strutture base): ✓ COMPLETATO (21 test)
- **Modulo 2** (setup partita): ✓ COMPLETATO (15 test)
- **Modulo 3** (motore di gioco — 4 fasi del turno): ✓ COMPLETATO (30 test)
- **Modulo 4** (sdadata e fine partita): ✓ COMPLETATO (19 test)
- **Test integrazione end-to-end**: ✓ COMPLETATO (6 test)
- **Modulo 5a** (encoding observation): ✓ COMPLETATO (16 test)
- **Modulo 5b** (action space e masking): ✓ COMPLETATO (16 test)
- **Modulo 5c** (environment Gymnasium): ✓ COMPLETATO (13 test)
- **Modulo 6** (training MaskablePPO su Colab): ✓ COMPLETATO

**Test totali**: 136 ✓

**Il progetto è completo end-to-end.** Puoi addestrare il bot.

## Specifica

La fonte di verità del regolamento è il file `risiko_specifica_v1.2.md` (separato).

## Setup

Python 3.11+ richiesto.

```bash
pip install -r requirements.txt
```

## Test (sul tuo PC)

```powershell
python tests\test_data.py              # 21 test (Modulo 1)
python tests\test_setup.py             # 15 test (Modulo 2)
python tests\test_motore.py            # 30 test (Modulo 3)
python tests\test_sdadata.py           # 19 test (Modulo 4)
python tests\test_partita_completa.py  #  6 test (integrazione)
python tests\test_encoding.py          # 16 test (Modulo 5a)
python tests\test_azioni.py            # 16 test (Modulo 5b)
python tests\test_env.py               # 13 test (Modulo 5c)
```

Totale atteso: **136 test passati**.

## Training del bot

Il training va eseguito su **Colab Pro** (GPU T4). Vedi `GUIDA_TRAINING.md` per le istruzioni passo-passo.

In sintesi:
1. Push del progetto su GitHub
2. Apri `notebooks/train.ipynb` su Colab
3. Esegui le celle in ordine
4. Ottieni i checkpoint su Google Drive

## Valutazione locale del bot

Per testare un modello sul tuo PC (richiede `sb3-contrib` e `torch`):

```powershell
python scripts\valuta_bot.py percorso\al\modello.zip --n_partite 100
```

## Uso dell'environment (esempio)

```python
from risiko_env import RisikoEnv
import numpy as np

env = RisikoEnv(seed=42, bot_color="BLU")
obs, info = env.reset()

terminated = False
while not terminated:
    mask = info["action_mask"]
    legali = np.where(mask)[0]
    azione = np.random.choice(legali)
    obs, reward, terminated, truncated, info = env.step(int(azione))

print(f"Vincitore: {info['vincitore']}, reward bot: {reward}")
```

## Observation & Action Space

**Observation**: 318 float32 (mappa, obiettivo proprio, carte, avversari pubblici, controllo continenti, fase, scarti).

**Action space**: Discrete(1765). Le azioni legali variano per sotto-fase, indicate dalla `action_mask` in `info`.

**Sotto-fasi del turno bot**: tris → rinforzo (× num armate) → attacco/continua/quantità_conquista → spostamento/quantità_spostamento.

## Reward

Sparso a fine partita:
- **+1.0** vittoria
- **+0.3** secondo posto
- **-0.3** terzo posto
- **-1.0** quarto posto / eliminato

## Struttura del progetto

```
risiko-rl/
├── README.md
├── GUIDA_TRAINING.md
├── requirements.txt
├── risiko_env/
│   ├── __init__.py
│   ├── data.py              # Modulo 1
│   ├── stato.py             # Modulo 1
│   ├── setup.py             # Modulo 2
│   ├── combattimento.py     # Modulo 3
│   ├── obiettivi.py         # Modulo 3
│   ├── motore.py            # Modulo 3
│   ├── sdadata.py           # Modulo 4
│   ├── encoding.py          # Modulo 5a
│   ├── azioni.py            # Modulo 5b
│   └── env.py               # Modulo 5c (RisikoEnv)
├── tests/
│   ├── test_data.py
│   ├── test_setup.py
│   ├── test_motore.py
│   ├── test_sdadata.py
│   ├── test_partita_completa.py
│   ├── test_encoding.py
│   ├── test_azioni.py
│   └── test_env.py
├── notebooks/
│   └── train.ipynb          # Modulo 6: training su Colab Pro
└── scripts/
    └── valuta_bot.py         # Valutazione standalone di un modello
```

## Strategia di training

- **Avversari iniziali:** bot random integrato nell'environment
- **Algoritmo:** MaskablePPO (sb3-contrib) con MlpPolicy
- **Hardware:** GPU T4 su Colab Pro
- **Durata prima sessione:** 1M timesteps (~30-60 min)
- **Self-play vero:** miglioramento futuro (sostituire bot random con bot RL)

## Aspettative win rate (baseline random = 25%)

| Step di training | Win rate atteso |
|---|---|
| 100k | 25-30% |
| 500k | 30-40% |
| 1M | **35-50%** |
| 5M | 50-65% |
| 10M+ | 65-80% |
