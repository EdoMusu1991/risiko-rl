# Roadmap completa: dal niente a un'AI forte di RisiKo

Documento sintesi del progetto. Racconta cosa abbiamo costruito, dove siamo oggi, e come arrivare a un bot competitivo contro umani.

Aggiornato al 29 aprile 2026.

---

## Indice

1. [Punto di partenza e obiettivo finale](#1-punto-di-partenza-e-obiettivo-finale)
2. [Cosa abbiamo fatto finora](#2-cosa-abbiamo-fatto-finora)
3. [Dove siamo oggi](#3-dove-siamo-oggi)
4. [Roadmap futura](#4-roadmap-futura)
5. [Tempistiche realistiche](#5-tempistiche-realistiche)
6. [Decisioni tecniche prese](#6-decisioni-tecniche-prese)
7. [Decisioni rimandate](#7-decisioni-rimandate)
8. [Glossario](#8-glossario)

---

## 1. Punto di partenza e obiettivo finale

### Da dove siamo partiti

All'inizio del progetto avevi:
- Un'app web "RisiKo Indovina gli Obiettivi" (Spring Boot + Angular + MySQL su Railway)
- 16 obiettivi mappati nel database
- 120 replay di partite Blitzwar (in formato non-JSON, da parsare)
- Esperienza fullstack ma non in RL

Volevi un bot che giocasse RisiKo contro umani in modo **strategicamente intelligente**, non meccanicamente.

### Definizione di "AI forte"

L'obiettivo non è un bot perfetto (impossibile senza dataset enormi tipo CICERO), ma un bot che:

1. **Conosce le regole** e le applica correttamente
2. **Gioca strategicamente**: usa tris al momento giusto, mira a continenti, gestisce confini
3. **Si adatta agli avversari**: capisce chi è aggressivo e chi difensivo, e reagisce
4. **Pensa "balance of power"**: sa di non diventare il bersaglio comune
5. **Batte i bot Blitzwar** stabilmente (i bot dell'app online dove giochi)
6. **Compete con umani amatoriali** (50%+ win rate contro giocatori medi)

L'obiettivo realistico in 4-6 mesi è il **livello 5**. Il livello 6 è possibile ma richiede dati umani di alta qualità.

---

## 2. Cosa abbiamo fatto finora

Ti racconto in ordine cronologico.

### Fase 0 — Decisioni strategiche (giorno 1)

Prima di scrivere una riga di codice, abbiamo deciso:

**Cosa abbandonare:**
- Dataset umano (i 120 replay non bastano per training, sarebbero stati behavior cloning di scarsa qualità)
- Porting del codice Java esistente (riscrittura da zero in Python è più rapida)
- Bot Java come avversari iniziali (troppa complessità di integrazione)

**Cosa scegliere:**
- **Self-play puro**: il bot gioca contro avversari random, poi contro versioni precedenti di sé stesso
- **Python da zero** con Gymnasium (interfaccia RL standard)
- **Reward sparso a fine partita**: +1 vittoria, +0.3 secondo, -0.3 terzo, -1 quarto. Il modo più puro di insegnare.
- **Action space atomico**: il bot decide solo le scelte strategiche (chi attaccare, dove rinforzare), il motore gestisce i dettagli (lanci di dadi, ecc.)
- **Training su Colab Pro**: GPU T4, ~100 compute units al mese

### Fase 1 — Estrazione del regolamento

Abbiamo formalizzato il regolamento del torneo italiano in un documento `risiko_specifica_v1.2.md`. Punti chiave:

- 4 giocatori (BLU, ROSSO, VERDE, GIALLO) in ordine di mano
- 42 territori, 6 continenti con bonus
- 16 obiettivi (20-23 territori target)
- Distribuzione iniziale: BLU/ROSSO 10 territori + 20 carri, VERDE/GIALLO 11 territori + 19 carri (tutti 30 armate totali)
- 4 fasi del turno: tris/rinforzi, attacchi, spostamento, pesca carta
- Sdadata obbligatoria dal round 35 (Giallo) / 36 (altri) con soglie crescenti
- Cap di sicurezza al round 60
- Vincitore in cascata: punti in obiettivo → punti fuori obiettivo → ordine di mano inverso (Giallo > Verde > Rosso > Blu)
- Cap 130 carri totali per giocatore

Questa specifica è la **fonte di verità**. Tutto il codice deve aderire.

### Fase 2 — Costruzione del simulatore (Moduli 1-5c)

Abbiamo costruito il simulatore in **5 moduli incrementali**, con test a ogni step:

**Modulo 1 (21 test)** — Dati statici e strutture:
- 42 territori, adiacenze, continenti, 16 obiettivi
- Classi `Carta`, `TerritorioStato`, `Giocatore`, `StatoPartita`

**Modulo 2 (15 test)** — Setup partita:
- Distribuzione territori con vincolo continentale (no continenti monocolore in setup)
- Piazzamento iniziale armate

**Modulo 3 (30 test)** — Motore di gioco:
- Combattimento con regola "parità → difensore"
- 4 fasi del turno
- Pesca carta su conquista
- Determinazione vincitore in cascata

**Modulo 4 (19 test)** — Sdadata e fine partita:
- Sdadata obbligatoria dal round 35-36 con soglie crescenti
- Cap di sicurezza round 60
- Verifica statistica: probabilità sdadata 0.171 vs 0.167 atteso

**Test integrazione (6 test)**:
- Partite end-to-end con bot random
- ~100 partite/sec single-thread

**Modulo 5a (16 test)** — Encoding observation:
- Stato → vettore 318 float (poi diventato 330 con Stage A)
- Privacy verificata: bot non vede obiettivi/carte specifiche degli avversari

**Modulo 5b (16 test)** — Action space:
- Discrete(1765) unificato
- Maschere dinamiche per sotto-fase

**Modulo 5c (13 test)** — Environment Gymnasium:
- State machine con 7 sotto-fasi
- Reward sparso a fine partita
- Bot avversari random integrati

A questo punto: **136 test verdi**, simulatore completo.

### Fase 3 — Primo training (training v1)

Lanciato su Colab Pro, GPU T4, 1M timesteps, ~33 minuti.

**Risultato**: win rate 12% (sotto il random teorico di 25%).

Diagnosi: il bot stava facendo deliberatamente peggio del random.

### Fase 4 — Diagnosi e fix critico

Abbiamo fatto un **sanity check del bot random** in tutte le posizioni.

Risultato sconvolgente: bot random come BLU vinceva il **5%**, come GIALLO il 37%, ROSSO 27%, VERDE 31%. Asimmetria gravissima.

Indagando: **i 3 avversari usavano un'euristica smart** (filtro ratio ≥ 1.5, 5 attacchi calcolati, spostamento 50%), mentre il bot RL random sceglieva tra 1765 azioni atomiche senza euristica. Asimmetria strutturale: il bot RL non poteva vincere nemmeno se avesse imparato perfettamente.

**Fix applicati (training v2)**:
1. Avversari resi completamente random (uniforme) — eliminata l'asimmetria
2. Reward shaping minimo: +0.001 per territorio conquistato, -0.0005 per perso
3. Iperparametri rivisti: lr da 3e-4 a 1e-4, n_steps da 512 a 2048, ent_coef da 0.01 a 0.05

**Risultato training v2**: win rate 24%, baseline simmetrica raggiunta. Gap eliminato.

### Fase 5 — Training v3 (in corso)

5M timesteps con i nuovi iperparametri. Primi numeri intermedi a 1M:
- `ep_rew_mean`: -0.272 (in salita da -0.547)
- Win rate atteso a 5M: 30-45%

### Fase 6 — Costruzione del toolkit

Mentre il training girava, abbiamo costruito strumenti per analizzare i risultati futuri:

- **`valuta_completo.py`** — Eval con CI Wilson, distribuzione posizioni, tracking best model, p-value
- **`visualizza_partita.py`** — Replay turno-per-turno con colori ANSI: tris, rinforzi, attacchi, spostamenti, carte, sdadata
- **`verifica_integrita.py`** — Stress test 1000 partite con 6 check di sanità
- **`confronta_partite.py`** — Matched-pairs A vs B, test McNemar
- **`heatmap_territori.py`** — Heatmap testuale di dove il bot conquista
- **`auto_eval_durante_training.py`** — Monitor live durante il training

### Fase 7 — Stage A (opponent embedding)

Aggiunte 12 feature alla observation (4 per ogni avversario, calcolate sulle ultime 8 mosse):
- Aggressività (% turni con almeno un attacco)
- Focus su di me (% attacchi diretti contro POV)
- Risk tolerance (ratio armate medio)
- Expansion rate (territori conquistati per turno)

Da 318 a **330 feature**. La rete impara da sola a profilare gli avversari.

Code pronto, **da trainare quando v3 finisce**.

### Fase 8 — Refactoring e documentazione

`env.py` da 1054 a 887 righe, splittato in 3 moduli specializzati:
- `bot_random.py` (logica avversari)
- `opponent_tracker.py` (storia mosse)
- `event_log.py` (logger eventi)

Aggiunti test di regressione (4 test) e 3 documenti:
- `CAPIRE_TRAINING.md` — Spiegone PPO e metriche
- `ARCHITETTURA.md` — Doc tecnica del codice
- `GUIDA_TRAINING.md` — Setup Colab passo-passo

---

## 3. Dove siamo oggi

**Codice**:
- Simulatore completo (Moduli 1-5c)
- Stage A pronto (non ancora trainato)
- 7 tool di analisi
- 148 test verdi
- 3 documenti tecnici + roadmap (questo)

**Training**:
- v1 (broken): 12% win rate — abbandonato
- v2 (fix asimmetria, vecchi iper): 24% win rate
- **v3 (nuovi iper, 5M step): in corso**

**Aspettative training v3**: 30-45% win rate.

**Compute residuo**: ~85 unità su 100 mensili.

---

## 4. Roadmap futura

5 stage incrementali. Ognuno aggiunge un livello di "intelligenza".

### Stage A — Opponent embedding 🔴 v1 fallita / 🟡 v2 (Stage A2) in test

**Stage A v1 (FALLITA)**: 4 feature × 3 avversari (aggressivita, focus_su_di_me, risk_tolerance, expansion_rate). Win rate 19% [CI 16-23] vs baseline 29%. CI non sovrapposti.

Diagnosi (verificata con `scripts/debug_stage_a.py`):
- `risk_tolerance` era un proxy fisso (0.4 quando l'avversario conquista, 0 altrimenti)
- `aggressivita`, `focus_su_di_me`, `expansion_rate` dipendevano dal SUCCESSO degli attacchi, non dal tentativo. Avversario aggressivo che fallisce = avversario passivo (indistinguibile)
- Tutte le feature nel range [0, 0.5], spazio sotto-utilizzato

**Stage A2 (in test)**: 8 feature × 3 avversari (24 totali). Feature di stato + storia leggera:
1-3. territori, armate, continenti (forza)
4-6. confini con me, armate sui confini, miei territori minacciati (vicinanza/minaccia)
7-8. conquiste recenti, perdite recenti (cambio dinamico)

Observation 318 → 342. Verificato con debug script: feature usano l'intero range [0, 1] e discriminano bene fra avversari diversi.

**Cosa fa**: Il bot vede statistiche sugli avversari (aggressività, focus, risk, expansion) e impara a profilarli.

**Cosa cambia nel comportamento**: Dovrebbe iniziare a riconoscere "questo nemico è aggressivo, devo rinforzare i miei confini" o "quello è passivo, posso permettermi di espandermi".

**Implementazione**: 12 feature in observation (318 → 330).

**Costo**: 1 training da 5M-10M (~3-6 ore di Colab).

**Win rate atteso**: 35-50% (vs ~25% baseline simmetrica).

---

### LEZIONE APPRESA: il "best model" non è il "final model"

Durante il training v4.1-test abbiamo scoperto una verità controintuitiva:

```
1M step → 30% win rate (bot vincente)
3M step → 19% win rate (bot conservativo, sopravvive ma non vince)
```

Il bot a 3M ha imparato a **aggirare** la penalty -0.001 (armate >= 125) tenendosi a 122 armate. Ha smesso di rischiare per chiudere le partite. Reward hacking.

**Regole che applichiamo da ora in avanti**:

1. **`MaskableEvalCallback`** durante ogni training: salva automaticamente il best model basato su reward medio in 20 partite di valutazione (ogni 100k step).

2. **Mai valutare solo il modello finale**. Sempre confronto fra `best_model.zip` (auto-salvato) e `risiko_bot_finale.zip` (ultimo step).

3. **Penalty graduali**, non a soglia secca. Una penalty `-0.001 if armate >= 125` insegna "stai a 124". Una penalty `-0.0005 * (armate - 100) / 30` cresce continuamente.

4. **Replay sano è criterio non negoziabile**. Win rate alto MA stallo cristallizzato = scartiamo.

---

### Stage B — Doppia finestra temporale

**Cosa fa**: Il bot ricorda cosa hanno fatto gli avversari NON solo nelle ultime 8 mosse (presente) ma anche nelle ultime 20 (memoria di lungo termine).

**Esempio pratico**: Al turno 3, ROSSO ti attacca senza motivo. Al turno 18, sai ancora che ROSSO ti aveva tradito una volta, e gli dai meno fiducia.

**Implementazione tecnica**:
- Espandere `OpponentTracker` per mantenere 2 buffer separati
- Aggiungere altre 12 feature (recente + memoria)
- Da 330 a 342 feature

**Costo**: 1 settimana di codice + 1 training.

**Win rate atteso**: +3-5% rispetto a Stage A.

---

### Stage C — Balance of power reward

**Cosa fa**: Reward shaping che penalizza il bot quando è "leader visibile senza coperture". Insegna a non diventare il bersaglio comune dei 3 avversari.

**Esempio pratico**: Hai conquistato Sud America, sei in testa per territori. Penalty per ogni confine scoperto verso un avversario. Il bot impara a coprire o a non espandersi troppo visibilmente.

**Implementazione tecnica**:
- Reward shaping aggiuntivo nell'env:
  - `-0.005 * confini_scoperti` se sei leader e non in continente completato
  - `+0.002` se mantieni profilo basso (non attacchi quando sei leader)

**Costo**: pochi giorni di codice + 1 training di tuning (i pesi sono delicati).

**Win rate atteso**: +5-10% rispetto a Stage B.

A questo punto siamo a circa **45-65%** di win rate vs random. Il bot è **chiaramente intelligente**, fa scelte sensate, sopravvive bene, conquista continenti. Rimane debole contro umani veri perché non sa giocare con stili specifici.

---

### Stage D — League play (Population-Based Training)

**Cosa fa**: Invece di un bot che si addestra contro 3 random, addestriamo **8-16 policy in parallelo** con stili diversi:
- Policy "aggressiva" (alta entropia, attacca spesso)
- Policy "difensiva" (gioca conservativo, accumula armate)
- Policy "espansiva" (mira a continenti)
- Policy "opportunista" (cambia stile in base allo stato)
- Ecc.

Ogni policy si addestra contro le altre. Periodicamente, le policy "deboli" vengono sostituite con varianti delle "forti" (con piccole mutazioni di iperparametri).

Il bot principale impara a **battere stili diversi**, non solo random. È la tecnica di AlphaStar (StarCraft) e CICERO (Diplomacy), ridotta all'osso.

**Implementazione tecnica**:
- Framework custom di Population-Based Training (PBT)
- Pool di N policy con file `.zip` separati
- Matchmaking interno: ogni partita di training, scegli 1 policy da addestrare + 3 dal pool come avversari
- Snapshot periodici, ranking ELO interno
- Replacement: ogni 1M step, le 25% peggiori vengono sostituite con varianti delle migliori

**Costo**: ~1 mese di codice + 5-10x più compute training rispetto a Stage A-C.

**Win rate atteso**: 65-75% vs random. Più importante: **robustezza contro stili sconosciuti**. Probabilmente batte i bot Blitzwar in modo netto.

---

### Stage E — Behavior cloning + piKL leggero

**Cosa fa**: Usiamo le 120 partite Blitzwar che hai. Le parsiamo, le augmentiamo (rotazioni, flip della mappa), arrivando a ~230k esempi di mosse umane.

Due step:
1. **Behavior cloning iniziale**: il bot inizia imitando gli umani (anziché partire random). Già a step 0 gioca "in modo umano". Poi raffina con self-play.
2. **Vincolo piKL leggero**: durante il self-play successivo, aggiungiamo un termine KL-divergence che mantiene la policy "vicina" allo stile umano. Evita che il bot evolva strategie aliene che funzionano in self-play ma falliscono contro umani.

Tecnica di **CICERO** (Meta AI, Diplomacy), versione semplificata.

**Implementazione tecnica**:
- Parser dei replay Blitzwar (formato da identificare)
- Pipeline di augmentation
- Pretraining della policy con cross-entropy loss
- Wrapper PPO con KL-regularizer verso una "behavior policy" frozen

**Costo**: 2 mesi di lavoro (parser è il pezzo lungo) + compute paragonabile a Stage D.

**Win rate atteso vs umani amatoriali**: 50-60%. Il bot inizia a essere **non distinguibile** da un giocatore umano funzionale, e a volte vince contro di loro.

---

## 5. Tempistiche realistiche

Considerando che lavori part-time sul progetto, con un ritmo realistico:

| Stage | Tempo dall'inizio | Win rate atteso vs random | Win rate vs umani | Compute |
|---|---|---|---|---|
| Baseline (oggi) | 0 | 24-45% | <5% | basso |
| Stage A | +1-2 settimane | 35-50% | <10% | basso |
| Stage B | +3-4 settimane | 40-55% | 10-20% | basso |
| Stage C | +5-6 settimane | 50-65% | 20-30% | medio |
| Stage D | +2-3 mesi | 65-75% | 30-45% | alto |
| Stage E | +4-6 mesi | 75-85% | 50-60% | alto |

**"Tempo"** include: codice + training + tuning + debug. Aggiungi 30-50% di buffer per imprevisti (è la regola d'oro del software).

---

## 6. Decisioni tecniche prese

Per memoria storica, ecco le scelte che abbiamo fatto e perché.

### Self-play vs imitation learning

**Scelto**: self-play puro per la baseline.
**Perché**: 120 partite Blitzwar sono troppo poche per behavior cloning serio (servirebbero 100k+ partite). Il valore è nello Stage E, non all'inizio.

### Reward sparso vs dense

**Scelto**: reward sparso (terminale ±1) + shaping minimo (territori ±0.001).
**Perché**: shaping aggressivo distorce la policy verso "guadagna territori", che non è la stessa cosa che "vincere" (puoi avere molti territori ma fallire l'obiettivo). Lo shaping minimo dà solo un "warm signal" alla rete.

### Action space atomico vs gerarchico

**Scelto**: spazio Discreto unificato (1765) con maschere dinamiche.
**Perché**: PPO standard non supporta nativamente azioni gerarchiche. Lo spazio unificato è più semplice e compatibile con MaskablePPO. La rete impara cosa è valido in quale fase.

### Bot avversari random vs euristici

**Scelto**: random uniforme, dopo aver scoperto il bug dell'asimmetria.
**Perché**: il bot RL parte random, deve poter competere a parità. Avversari smart introducono asimmetria distruttiva. Self-play vero (Stage D) sostituirà il random.

### Single-bot vs multi-agent

**Scelto**: single-bot (1 RL + 3 random). Multi-agent vero solo allo Stage D.
**Perché**: multi-agent simultaneo è 10× più complicato. Per la baseline, il single-bot è sufficiente.

### MaskablePPO vs altri algoritmi

**Scelto**: MaskablePPO (sb3-contrib).
**Perché**: gestisce nativamente action masking (1765 azioni di cui solo poche legali per fase). PPO è lo standard. SAC e A3C non supportano bene action masking.

### Colab Pro vs cluster locale

**Scelto**: Colab Pro.
**Perché**: hai un PC normale, niente GPU dedicata. Colab Pro a 11€/mese ti dà T4 e 100 compute units. Sufficiente per Stage A-C, parzialmente per D-E.

---

## 7. Decisioni rimandate

Cose che abbiamo discusso ma non abbiamo ancora deciso. Le scegliamo quando arriverà il momento.

### Quando passare a self-play vero

Possiamo farlo allo Stage D (league play) o anche prima, sostituendo i bot random con copie congelate del bot RL stesso. Decisione futura.

### Architettura della rete neurale

Adesso usiamo `MlpPolicy` standard (~250-300k parametri). Considereremo `LSTM-based` o `Transformer` se la baseline satura troppo presto. Probabile dallo Stage D.

### Curriculum learning

Iniziare con avversari deboli e gradualmente aumentare la difficoltà. Possibile boost di sample efficiency. Decisione allo Stage D.

### Auto-tuning iperparametri

Optuna o simili. Costoso in compute. Da considerare quando la baseline è solida.

### Distillazione

Quando avremo un bot "campione" da Stage E, possiamo distillarlo in una rete più piccola e veloce per inferenza in produzione (es. integrazione nell'app web). Decisione finale.

---

## 8. Glossario

Termini che ho usato nel doc, in ordine alfabetico.

**Action masking**: meccanismo che dice alla rete quali azioni sono legali in un dato momento. La rete apprende a non scegliere quelle illegali.

**Behavior cloning**: tecnica di RL in cui il bot inizializza la policy imitando un dataset di mosse umane. Punto di partenza, non destinazione finale.

**CICERO**: bot di Meta AI per Diplomacy (2022). Combina RL + behavior cloning + linguaggio naturale. Riferimento di stato dell'arte in giochi multi-player con interazione.

**Compute units**: unità di "tempo CPU/GPU" su Colab Pro. 100 al mese, ~1.2 unità/ora su T4.

**Critic**: rete neurale che stima "quanto vale" uno stato. Parte di PPO. Aiuta l'actor a capire se una mossa migliora o peggiora la situazione.

**ELO**: sistema di ranking che assegna a ogni giocatore un punteggio numerico, aggiornato dopo ogni partita. Usato in scacchi e in gran parte dei giochi competitivi.

**Entropia (entropy)**: misura quanto "incerta" è la policy. Alta entropia = esplora molte mosse. Bassa entropia = sfrutta una strategia. Il `ent_coef` controlla il bilanciamento.

**Episode**: una partita completa (da reset a fine).

**Explained variance**: quanto bene il critic prevede i reward. 1 = perfetto, 0 = completamente cieco.

**Gymnasium**: libreria Python standard per environment RL. Successore di OpenAI Gym. Definisce l'interfaccia `reset()`, `step()`, ecc.

**Imitation learning**: famiglia di tecniche che includono behavior cloning. Imparare imitando esempi.

**KL-divergence**: misura quanto due distribuzioni di probabilità sono diverse. Usata in PPO per limitare quanto la policy nuova si discosta dalla vecchia.

**League play**: training simultaneo di molte policy con stili diversi che si sfidano a vicenda. Inventato da AlphaStar.

**Maskable PPO**: variante di PPO che supporta action masking nativamente. Disponibile in `sb3-contrib`.

**piKL**: tecnica che combina policy iteration con KL-divergence verso una behavior policy umana. Usata in CICERO per mantenere il bot "umano-leggibile".

**Policy**: la "rete neurale di decisione" del bot. Funzione che prende osservazione e sputa distribuzione di probabilità sulle azioni.

**Population-Based Training (PBT)**: tecnica che addestra una popolazione di policy in parallelo, con replacement periodico delle peggiori. Stage D del nostro progetto.

**PPO (Proximal Policy Optimization)**: algoritmo RL standard inventato da OpenAI nel 2017. Usato in ChatGPT, AlphaStar, e molti altri. Cuore del nostro training.

**Replay buffer**: dataset di mosse passate accumulate durante il training. PPO non lo usa (è on-policy), ma altri algoritmi sì (DQN, SAC).

**Reward shaping**: aggiungere reward intermedi piccoli per guidare l'apprendimento. Pericoloso: distorce la policy verso il proxy invece che verso l'obiettivo vero.

**Self-play**: il bot gioca contro versioni di sé stesso. Tecnica chiave per superare la limitazione "non ho dati umani".

**Stage A/B/C/D/E**: i 5 livelli incrementali della nostra roadmap (vedi sezione 4).

**State machine**: schema a stati finiti che gestisce il flusso del turno. Il nostro env ne ha una con 7 sotto-fasi.

**Timestep**: una singola decisione del bot (= una chiamata `step()`). 1M timesteps è il default per il primo training.

**Wilson confidence interval**: intervallo di confidenza per proporzioni, più accurato del normale quando p è vicino a 0/1 o n è piccolo. Usato in `valuta_completo.py`.

---

## Conclusione

Quando questo documento è stato scritto, eravamo in una fase ben definita:
- Simulatore solido con 148 test
- Toolkit di analisi professionale
- Stage A pronto in coda
- Training v3 in corso (atteso 30-45% win rate)

I prossimi passi concreti, in ordine:

1. **Aspettare risultati training v3** (ore)
2. Se baseline ≥ 30%: lanciare **Stage A** (settimana)
3. Continuare con **Stage B** e **Stage C** in successione (mese)
4. Quando il bot fa 50%+ vs random: pensare seriamente a **Stage D** (mese 2-3)
5. **Stage E** per il bot finale (mese 4-6)

Nessuno di questi step è "garantito di funzionare". RL è fragile, gli iperparametri sono sensibili, i bug sono comuni. Aggiungi 30-50% di buffer alle stime.

Ma il progetto è **ben strutturato** per arrivare lontano. Hai:
- Codice modulare e testato
- Toolkit di analisi
- Documentazione tecnica
- Roadmap chiara
- Conoscenza tecnica dell'algoritmo

Quello che manca è il **tempo di training** e la **pazienza di iterare**. Sono cose che si comprano solo coi mesi.

Buona strada.
