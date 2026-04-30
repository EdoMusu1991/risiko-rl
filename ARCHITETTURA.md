# Architettura tecnica del progetto

Questo documento descrive **come è strutturato il codice** del progetto, modulo per modulo. Utile per chi (te incluso, fra 6 mesi) deve mettere mano al simulatore o estenderlo.

Per **come usare** gli script: `README.md`.
Per **come funziona PPO**: `CAPIRE_TRAINING.md`.

---

## Albero del progetto

```
risiko-rl/
├── risiko_env/                    Pacchetto Python principale
│   ├── __init__.py               Esporta RisikoEnv come API pubblica
│   │
│   ├── data.py                   Costanti del gioco
│   ├── stato.py                  Modello dati (StatoPartita, Giocatore)
│   ├── setup.py                  Inizializzazione partita
│   ├── combattimento.py          Lancio dadi e risoluzione
│   ├── obiettivi.py              Determinazione vincitore
│   ├── motore.py                 Logica delle 4 fasi del turno
│   ├── sdadata.py                Sdadata e cap di sicurezza
│   │
│   ├── encoding.py               Stato → vettore observation
│   ├── azioni.py                 Action space e maschere
│   ├── env.py                    Gymnasium environment
│   │
│   ├── bot_random.py             Bot avversari (estratto da env.py)
│   ├── opponent_tracker.py       Storia mosse + profilo (estratto)
│   └── event_log.py              Logger eventi per visualizzatore
│
├── tests/                         Test (148 totali)
│   ├── test_data.py              21 test
│   ├── test_setup.py             15 test
│   ├── test_motore.py            30 test
│   ├── test_sdadata.py           19 test
│   ├── test_partita_completa.py   6 test (integrazione)
│   ├── test_encoding.py          16 test
│   ├── test_azioni.py            16 test
│   ├── test_env.py               13 test
│   ├── test_opponent_profile.py   8 test
│   └── test_regressione.py        4 test (smoke check + bot random)
│
├── scripts/                       Tool di analisi e training
│   ├── valuta_bot.py             Eval base
│   ├── valuta_completo.py        Eval con CI Wilson + tracking best
│   ├── visualizza_partita.py     Replay partita commentato
│   ├── verifica_integrita.py     Stress test env (1000 partite)
│   ├── confronta_partite.py      Matched-pairs fra 2 modelli
│   ├── heatmap_territori.py      Heatmap zone preferite dal bot
│   └── auto_eval_durante_training.py    Monitor live durante training
│
├── notebooks/
│   └── train.ipynb               Notebook Colab per training
│
├── README.md                      Manuale utente
├── CAPIRE_TRAINING.md             Spiegone PPO e metriche
├── ARCHITETTURA.md                Questo file
├── GUIDA_TRAINING.md              Setup Colab Pro step-by-step
└── requirements.txt               Dipendenze Python
```

---

## Layer 1: Modello dati (`stato.py`, `data.py`)

**`data.py`** contiene **costanti immutabili** del gioco:

- `TUTTI_TERRITORI` — 42 territori
- `ADIACENZE` — dict `territorio → lista_adiacenti`
- `CONTINENTI` — dict `nome → lista_territori`
- `BONUS_CONTINENTI` — dict `nome → bonus_armate`
- `OBIETTIVI` — 16 obiettivi (id → descrizione + funzione completamento)
- `COLORI_GIOCATORI` — `["BLU", "ROSSO", "VERDE", "GIALLO"]`
- `MAX_ARMATE_TOTALI = 130` — cap regolamento

Tutto **read-only**. Mai modificato a runtime.

**`stato.py`** definisce le classi mutabili:

- `Carta` — singola carta territoriale (territorio + simbolo)
- `TerritorioStato` — stato di un territorio: `proprietario`, `armate`
- `Giocatore` — `colore`, `obiettivo_id`, `carte`, `vivo`
- `StatoPartita` — stato globale: mappa, turno corrente, fase, ecc.

Nessuna logica di gioco qui dentro. Solo data classes.

---

## Layer 2: Motore di gioco (`motore.py`, `combattimento.py`, `obiettivi.py`, `sdadata.py`)

Funzioni **pure** che operano su `StatoPartita` e lo modificano. Tutte testate in isolamento.

**`combattimento.py`**: lancia i dadi, applica perdite. La regola "in pareggio vince il difensore" è qui.

**`motore.py`**: orchestrazione delle 4 fasi del turno:

1. `gioca_tris(stato, colore, scelta)` → bonus armate per tris
2. `piazza_rinforzi(stato, colore, distribuzione)` → posiziona armate
3. `esegui_attacco(stato, colore, da, verso, rng, fermati_dopo_lanci)` → lancia dadi
4. `applica_conquista(stato, colore, da, verso, quantita, esito, rng)` → trasferisce armate, gestisce eliminazione, controlla vittoria
5. `esegui_spostamento(stato, colore, da, verso, quantita)` → spostamento finale
6. `pesca_carta(stato, colore, rng)` → se conquistato, pesca

**`obiettivi.py`**: `ha_completato_obiettivo()`, `determina_vincitore()` con i 3 criteri in cascata (punti obj → punti fuori obj → ordine mano inverso).

**`sdadata.py`**: `gestisci_fine_turno()`. Implementa la sdadata obbligatoria dal round 35 (Giallo) / 36 (altri) con soglie crescenti, e il cap di sicurezza al round 60.

---

## Layer 3: Adapter RL (`encoding.py`, `azioni.py`)

Strato di traduzione tra il motore di gioco e gli algoritmi RL.

**`encoding.py`**: trasforma `StatoPartita` in un vettore numpy di **330 float**, dal punto di vista di un giocatore (POV).

Sezioni della observation:

| Sezione | Dimensione | Contenuto |
|---|---|---|
| Mappa | 252 | 42 territori × 6 feature (proprietà one-hot, armate, è in obj) |
| Obiettivo proprio | 16 | one-hot dell'obiettivo del POV |
| Carte proprie | 5 | conteggio per simbolo + totale |
| Avversari | 12 | per ognuno: territori, armate, carte, vivo |
| Continenti | 24 | chi controlla ogni continente (4×6) |
| Fase + turno | 6 | round normalizzato, fase one-hot, conquiste |
| Tris pubblici | 3 | scarti per simbolo (info pubblica) |
| **Opponent profile** | **12** | **Stage A: 4 feature × 3 avversari** |

Privacy: il bot vede solo le info che spettano a un giocatore reale. NON vede obiettivi/carte specifiche degli avversari.

**`azioni.py`**: action space e maschere.

L'action space è `Discrete(1765)` = max delle dimensioni delle 6 sotto-fasi del turno:

| Sotto-fase | NUM_AZIONI | Significato |
|---|---|---|
| TRIS | 11 | combinazioni di 0/1/2 tris da giocare |
| RINFORZO | 42 | territorio dove piazzare 1 armata |
| ATTACCO | 1765 | (da, verso) pair, oppure stop |
| CONTINUA | 2 | continua a tirare oppure stop |
| QUANTITA | 3 | min, mid, max armate da spostare |
| SPOSTAMENTO | 1765 | (da, verso) pair, oppure skip |

Le **maschere** dicono quali indici sono legali in una data sotto-fase. Calcolate dinamicamente in `azioni.py`. La rete apprende a ignorare gli indici fuori-fase.

---

## Layer 4: Environment Gymnasium (`env.py`)

`RisikoEnv(gym.Env)` orchestra tutto. Implementa l'interfaccia standard:

```python
env = RisikoEnv(seed=42, bot_color="BLU")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

### State machine interna

Il turno del bot è suddiviso in **7 sotto-fasi**, gestite da una macchina a stati:

```
TRIS → RINFORZO* → ATTACCO ⇄ CONTINUA → QUANTITA_CONQUISTA
                                  ↓
                            SPOSTAMENTO → QUANTITA_SPOSTAMENTO → fine_turno
```

`*` ripetuto N volte (uno per ogni armata da piazzare)

Ogni `step()` consuma una decisione del bot e fa avanzare la macchina. Quando la sotto-fase è `None`, il turno è finito e si passa agli avversari.

### Multi-agent semplificato

Il bot RL controlla 1 giocatore. Gli altri 3 sono **bot random** (`bot_random.py`).
Quando arriva il turno di un avversario, l'env lo gioca automaticamente senza chiedere al bot.

Self-play vero: futura evoluzione (Stage 4 della roadmap).

### Componenti modulari (post-refactoring)

`env.py` delega a 3 componenti:

- **`OpponentTracker`** (`opponent_tracker.py`): traccia le mosse degli avversari per Stage A. Mantiene un buffer di `MAX_STORIA_PER_COLORE = 50` mosse per colore. Espone `snapshot_pre_turno()` e `registra_mossa()`.

- **`EventLogger`** (`event_log.py`): se `log_eventi=True`, registra eventi nella partita per il visualizzatore. Quando `log_eventi=False` (default in training), è no-op.

- **`bot_random.gioca_turno_random()`** (`bot_random.py`): logica per i 3 avversari. Strategia uniforme (scelta random tra azioni legali, niente euristiche).

L'env espone proprietà legacy `_storia_mosse` e `_eventi` come facade verso i componenti, per backward-compat con tool esterni.

### Reward sparso + shaping

Reward = 0 durante la partita, eccetto:

- Reward terminale: `+1.0` (vince), `+0.3` (secondo), `-0.3` (terzo), `-1.0` (quarto/eliminato)
- **Reward shaping** (durante partita, magnitudo bassa per non sovrastare il terminale):
  - `+0.001` per ogni territorio conquistato
  - `-0.0005` per ogni territorio perso

Lo shaping aiuta a dare segnale alla rete in un ambiente con reward altrimenti molto sparso.

---

## Layer 5: Tool di analisi (`scripts/`)

Tutti i tool sono **stand-alone**: caricano un modello da `.zip` e lo testano.

| Script | Scopo | Quando usarlo |
|---|---|---|
| `valuta_bot.py` | Eval semplice | Veloce sanity check |
| `valuta_completo.py` | Eval approfondita | Confronto serio fra modelli |
| `visualizza_partita.py` | Replay commentato | Capire come gioca il bot |
| `verifica_integrita.py` | Stress test env | Dopo modifiche al simulatore |
| `confronta_partite.py` | Matched-pairs A vs B | Decidere se v2 batte v1 |
| `heatmap_territori.py` | Analisi spaziale | Capire strategia territoriale |
| `auto_eval_durante_training.py` | Monitor live | In parallelo al training |

---

## Convenzioni di codice

- **Lingua**: italiano per nomi di funzioni di gioco (`gioca_tris`, `piazza_rinforzi`, `esegui_attacco`), inglese per termini ML standard (`reset`, `step`, `policy`, `observation`).
- **Privacy bot**: tutto ciò che è privato di un giocatore (obiettivo, carte specifiche) NON deve finire nell'observation degli avversari. Test specifici in `test_encoding.py`.
- **Determinismo**: tutte le funzioni che usano randomness accettano un `random.Random` come parametro. Niente `random.choice` globale. Necessario per riproducibilità.
- **Test prima dei push**: ogni modifica al simulatore richiede `python tests/<modulo>.py`. CI/CD non c'è (ancora).

---

## Estensioni future (roadmap)

Documentate qui per memoria, non implementate.

### Stage A — Opponent embedding ✅

Implementato. Aggiunge 12 feature alle observation per profilare gli avversari.

### Stage B — Doppia finestra (memoria lunga)

Espandere `OpponentTracker` per mantenere 2 buffer:
- Recente: ultimi 8 turni (presente)
- Memoria: ultimi 20 turni (passato)

Modifica: `encoding.py` (DIM_OPPONENT_PROFILE 12 → 24), `OpponentTracker` (storia bipartita).

### Stage C — Balance of power reward

Aggiungere reward shaping che penalizza il bot quando è "leader visibile":

```python
if num_territori_bot == max_per_colore and not_in_continente_completato:
    reward -= 0.005 * confini_scoperti
```

Insegna a non diventare il bersaglio comune.

### Stage D — League play

Sostituire i 3 bot random con un pool di 8-16 policy diverse trainate in parallelo. Tecnica AlphaStar/CICERO. Richiede framework PBT custom.

### Stage E — Behavior cloning + piKL

Parsare 120 partite Blitzwar, augmentare a 230k esempi, behavior cloning della policy iniziale, vincolo KL durante self-play.

---

## Test

148 test totali, organizzati per modulo:

```
test_data           (21)  ─┐
test_setup          (15)  ├─  Layer 1-2: dati e motore
test_motore         (30)  │
test_sdadata        (19)  │
test_partita_completa (6) ─┘  (smoke test integrazione)

test_encoding       (16)  ─┐
test_azioni         (16)  ├─  Layer 3: adapter RL
test_env            (13)  │
test_opponent_profile (8) ─┘  (Stage A)

test_regressione    (4)       Smoke check modelli + bot random
```

Tutti i test sono **eseguibili in isolamento**: `python tests/test_X.py`. Niente pytest, niente fixtures complicate. Output: `TUTTI I N TEST PASSATI ✓` o lista falliti.

Lanciare tutti i test:

```powershell
foreach ($f in @("test_data","test_setup","test_motore","test_sdadata",
                 "test_partita_completa","test_encoding","test_azioni",
                 "test_env","test_opponent_profile","test_regressione")) {
    python "tests\$f.py"
}
```

---

## Performance

- **Simulazione**: ~500 partite/sec con bot random (Python single-thread)
- **Training**: ~500 fps su Colab T4 (8 envs paralleli, n_steps=2048)
- **1M timesteps** ≈ 30-35 min su T4
- **5M timesteps** ≈ 2.5-3 ore su T4

Bottleneck: la simulazione Python, non la GPU. La rete è piccola (250k-300k parametri).

---

## Dipendenze

Minime, intenzionalmente:

- `numpy>=1.26` — array, observation
- `gymnasium>=0.29` — interfaccia env

Per training (solo Colab):

- `sb3-contrib>=2.2` — MaskablePPO
- `torch>=2.0` — backend rete neurale

Niente pandas, scikit-learn, ecc. Il progetto è volutamente lean.
