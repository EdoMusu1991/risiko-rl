# risiko-rl

Simulatore RisiKo in Python (variante torneo italiano) come ambiente Gymnasium per addestramento di un bot RL.

## Documentazione

- **README.md** (questo file) — Manuale utente: come usare gli script
- **ROADMAP.md** — Tutto il percorso del progetto, dove siamo, dove andiamo
- **ARCHITETTURA.md** — Documentazione tecnica del codice
- **CAPIRE_TRAINING.md** — Spiegazione di PPO e come leggere le metriche
- **GUIDA_TRAINING.md** — Setup Colab Pro passo-passo

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

**Valutazione base** (win rate semplice):
```powershell
python scripts\valuta_bot.py percorso\al\modello.zip --n_partite 100
```

**Valutazione approfondita** (statistiche complete, intervalli di confidenza):
```powershell
python scripts\valuta_completo.py modello.zip
```

Misura win rate per ogni posizione (BLU/ROSSO/VERDE/GIALLO), distribuzione piazzamenti, statistiche di gioco (territori, armate, continenti, durata), con intervalli di confidenza Wilson al 95%.

**Solo posizione BLU** (più veloce, 4× meno tempo):
```powershell
python scripts\valuta_completo.py modello.zip --solo_blu --n_partite 200
```

**Confronto fra modelli** (con test di significatività statistica):
```powershell
python scripts\valuta_completo.py modello_v1.zip --confronta modello_v2.zip
```

Output con p-value: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` non significativo.

**Salvataggio CSV** per tracciare progresso nel tempo:
```powershell
python scripts\valuta_completo.py modello.zip --csv risultati.csv
```

**Tracking del miglior modello** (confronta automaticamente con il best precedente):
```powershell
python scripts\valuta_completo.py modello.zip --track_best
```

Salva un file `best_model.json` col modello migliore visto. Lanciando con `--track_best` su un nuovo modello, ti dice se è meglio o peggio del best (con p-value).

## Stress test dell'environment

Per verificare l'integrità dell'env (utile dopo modifiche al simulatore):

```powershell
python scripts\verifica_integrita.py --n_partite 1000
```

Gioca 1000 partite con bot random e verifica:
- Distribuzione vincitori bilanciata (±10% dal 25%)
- Durate partite plausibili (300-700 step)
- Cap 130 mai superato
- Distribuzione obiettivi uniforme (deviazione max 30%)
- Nessuna eccezione, nessuna partita troncata

## Confronto fra modelli partita-per-partita

Più informativo del solo win rate aggregato:

```powershell
python scripts\confronta_partite.py modello_v1.zip modello_v2.zip
```

Mostra per ogni seed chi vince con A e chi con B, tabella di contingenza, e test McNemar (matched-pairs) per significatività statistica.

## Heatmap territoriale del bot

Per capire dove il bot privilegia conquistare:

```powershell
python scripts\heatmap_territori.py modello.zip --n_partite 100
```

Heatmap testuale con barre per ogni territorio e continente. Vedi se il bot ha imparato strategie di continente o gioca random.

## Monitor automatico durante il training

Lanci questo in parallelo al training Colab (in una console separata o in una nuova cella) per vedere la progressione in tempo reale:

```powershell
python scripts\auto_eval_durante_training.py /percorso/cartella/checkpoint --n_partite 30
```

Sorveglia la cartella checkpoint, valuta automaticamente ogni nuovo file, salva progressione in CSV. Output con grafico testuale dell'evoluzione del win rate.

## Visualizzare una partita giocata dal bot

Per vedere come gioca davvero il bot, partita commentata turno-per-turno:

```powershell
python scripts\visualizza_partita.py modello.zip --seed 42
```

Mostra ogni attacco (chi attacca chi, armate prima/dopo, esito), cosa fanno gli avversari, e lo stato della partita ad intervalli regolari.

**Solo eventi del bot** (più conciso):
```powershell
python scripts\visualizza_partita.py modello.zip --solo_bot
```

**Cambia colore del bot**:
```powershell
python scripts\visualizza_partita.py modello.zip --bot_color GIALLO --seed 7
```

**Senza modello** (test col bot random, utile per verificare lo script):
```powershell
python scripts\visualizza_partita.py --random --seed 42
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

**Per capire cosa succede durante il training**, leggi `CAPIRE_TRAINING.md`: spiega in italiano cos'è PPO, come leggere le metriche di Colab, cosa significano `ep_rew_mean`, `entropy_loss`, `explained_variance`, e come capire se un training sta andando bene o male.

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
