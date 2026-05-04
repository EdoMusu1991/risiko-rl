# PROGETTO RISIKO ALPHA — Specifica v2

**Data**: Maggio 2026
**Stato**: Decisione presa, in attesa di partenza
**Direzione**: AlphaZero-style (MCTS + Neural Network) applicato a Risiko

---

## 1. Obiettivo

Costruire un bot di Risiko che giochi a livello "umano esperto" — definito come:
> Un giocatore esperto è chi gioca regolarmente Risiko, conosce il regolamento, capisce le 4 dinamiche di alto livello (statistica, equilibrio del tavolo, attacchi funzionali, uso strumentale degli avversari), e prende decisioni che riflettono planning a 5-10 turni avanti.

**NON è**: campione di tornei, livello AlphaStar/AlphaZero per scacchi.
**È**: un bot serio che batte 50%+ contro giocatori esperti in test 1 vs 3.

**Probabilità di successo stimata**: 35-50% sull'obiettivo principale.
(Era 40-60% prima delle critiche di ChatGPT. Abbassata per realismo: la complessità multi-player è maggiore di quanto stimassi inizialmente.)

---

## 2. Perché RL puro non basta

Le 5 sessioni di lavoro precedenti hanno prodotto questi dati:

| Approccio | Win rate vs random | Note |
|---|---|---|
| Baseline PPO 500k | **29%** [25-33%] | Migliore baseline ottenuta |
| PPO 3M (over-training) | 19% | Reward hacking |
| Stage A v1 (4 feat) | 19% | Feature mal progettate |
| Stage A2 (8 feat) | non testato | Cambio direzione |

**Conclusione operativa**: PPO+random ha un soffitto intorno al 27-29% e converge sempre lì indipendentemente da iperparametri o numero di step. Non scopre strategie psicologiche perché:
1. Random non reagisce → niente segnale di "non esporti"
2. Reward locale +1/-1 non vede catene causali a 10 turni
3. Lo spazio osservazionale non rappresenta "minacce future"

**Il salto di qualità richiesto** (planning, prevenzione, modellare gli altri) è strutturalmente fuori portata di PPO standard. Serve cambio di paradigma.

---

## 3. Architettura AlphaZero per Risiko

### 3.1 Componenti principali

```
                    ┌─────────────────────┐
                    │  Stato del gioco    │
                    │  (vettore 318)      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Rete Neurale      │
                    │  Policy + Value     │
                    └──────────┬──────────┘
                               │
                               │ (guida)
                               ▼
                    ┌─────────────────────┐
                    │       MCTS          │
                    │  (800 simulazioni)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Azione scelta     │
                    └─────────────────────┘
```

### 3.2 MCTS adattato

A ogni turno del bot:
1. **Inizializzazione**: stato corrente diventa root dell'albero
2. **Loop di simulazione** (800-1600 volte):
   - **Selezione**: scendi nell'albero scegliendo nodi con UCB modificato (PUCT)
   - **Espansione**: quando arrivi a un nodo non visitato, aggiungilo
   - **Valutazione**: usa la rete (o euristica nel prototipo) per stimare il valore
   - **Backup**: propaga il valore in alto nell'albero
3. **Decisione finale**: scegli l'azione visitata più volte dalla root

**Adattamenti specifici a Risiko**:
- **Action space ridotto**: l'env attuale ha azioni gerarchiche (fase tris → fase rinforzo → ecc.). MCTS lavora sulla decisione attiva, non su tutto lo spazio
- **Chance nodes**: i dadi del combattimento sono nodi di caso. MCTS standard fa "expectimax" su questi nodi
- **Multi-agent**: è un gioco a 4 giocatori. Gestiamo come "1 vs altri" usando il bot RL principale + euristica per gli avversari nelle simulazioni MCTS

### 3.3 Rete neurale (fase 2, da progettare)

Architettura base (rivedere dopo prototipo MCTS):

**Input**: vettore di stato 318 features (l'attuale encoding) + opzionali Stage A2 features
**Body**: 3-5 layers fully-connected con residual connections
**Heads**:
- **Policy head**: distribuzione su action space (usa action mask)
- **Value head**: scalare in [-1, 1], stima del risultato finale

**Loss**:
- Policy: cross-entropy fra policy della rete e visite di MCTS
- Value: MSE fra valore della rete e risultato finale partita

**NON useremo**: PPO. Andiamo direttamente con la formulazione AlphaZero.

### 3.4 Self-play loop

```
Iter 0: rete random → MCTS guidata da rete random
Per ogni iter N = 1, 2, 3, ...:
    Generate dataset:
        Per K partite:
            Bot vs Bot vs Bot vs Bot (4 copie della rete corrente)
            Per ogni mossa, registra (stato, policy MCTS, risultato finale)
    Train rete su dataset (mix nuovo + recente)
    Salva nuova rete in population
Periodicamente:
    Tournament fra reti per misurare progresso
```

### 3.5 Reward riformulato

**Dal binario al margine**:

Il reward attuale è +1 (vincita), +0.3 (2°), -0.3 (3°), -1 (4°). Lo cambiamo a:

```
reward_finale = (punti_propri - punti_max_avversari) / 100
clipped a [-1, +1]
```

Esempi:
- Vinco 60 pt vs 40 dell'avversario migliore → reward +0.20
- Vinco 80 vs 20 → reward +0.60
- Perdo 30 vs 70 → reward -0.40

**Motivo**: il bot impara non solo a vincere, ma a **massimizzare il margine**. Questo cattura "fare più punti degli altri" che hai descritto come regola 1.

**NB**: il reward usato da MCTS sarà questo. La rete value impara a stimare questo margine.

---

## 4. Roadmap dettagliata

### Mese 1 — MCTS prototipo (senza rete) — IN MODALITÀ 1V1

Mese 1 lavora in modalità ridotta 2 giocatori (1 vs 1). Niente diplomazia, niente kingmaking.
Motivo: in 4-player il valore non è stabile, MCTS può convergere a strategie nonsense.
In 1v1 il problema è ben definito e ti permette di capire VERAMENTE se MCTS funziona.
Estensione a 4-player nel Mese 2.

**Settimana 1**: Determinismo dell'env (PRIORITÀ ASSOLUTA)

Prima ancora del rollout, prima ancora di MCTS, l'env DEVE garantire:
```
stato + seed → evoluzione identica 100% delle volte
```

Lavori:
- Implementare `env.snapshot()` → cattura completa stato (deep copy, RNG interno incluso)
- Implementare `env.restore(snapshot)` → riporta env a stato precedente
- 4 test obbligatori:
  1. **Idempotenza**: snapshot → restore → snapshot identico
  2. **Determinismo**: stesso stato + stessa azione = stesso risultato
  3. **Replay**: stessa sequenza azioni da stesso snapshot = stesso esito
  4. **No side effects**: simulazioni non alterano env reale
- Misurare tempo snapshot/restore (target: <1ms, MCTS lo chiamerà migliaia di volte)
- Adattare `RisikoEnv` per modalità 1v1 (2 giocatori, condizioni vittoria appropriate)

Questi 4 test passano? Procediamo. Anche un solo test crepa? Debug finché non passa.

**Settimana 2**: Rollout policy + MCTS base in 1v1
- Rollout policy euristica (no random):
  - Attacca solo se win prob > 55-60%
  - Rinforza territori con nemici adiacenti
  - Evita attacchi che lasciano territori a 1 armata
- Implementare PUCT con la rollout policy
- Budget per mossa: target 1-2 secondi
- Iniziare con ~100 simulazioni, adattare in base al tempo

**Settimana 3**: Tuning MCTS in 1v1
- Misure: tempo per mossa, winrate vs random
- Sperimentare 100/200/300 simulazioni
- Tunare costanti UCB
- Test riproducibilità su 100 partite

**Settimana 4**: Validazione 1v1 + test "false positive"
- MCTS bot 1v1 vs random bot 1v1: vince 70%+? (atteso sì)
- MCTS bot 1v1 vs **baseline PPO 29% 1v1**: vince? (test critico per rischio 6)
- Decisione: MCTS è viable per Risiko? Sì/No/Forse

**Milestone Mese 1**: avere risposta SÌ/NO a "MCTS è viable per Risiko 1v1".

Solo se SÌ, procediamo al Mese 2.

### Mese 2 — Rete neurale

Solo se Mese 1 ha dato SÌ.

**Settimana 5**: Architettura rete
- Progetto policy+value head
- Loss function
- Training pipeline

**Settimana 6**: Generazione dataset
- Self-play con MCTS+rete random
- Salvataggio (stato, policy, valore) per ogni mossa
- Target: 10k partite generate

**Settimana 7**: Primo training
- Train rete sulle 10k partite
- Validazione: rete addestrata vs MCTS-only? Vince?

**Settimana 8**: Tournament + decisione
- Generazione 1 (rete v1) vs baseline 29%
- Generazione 1 vs MCTS-only
- Decidere: continuare iterazioni o tornare indietro

**Milestone Mese 2**: prima generazione AlphaZero validata o decisione di pivot.

### Mesi 3-4 — Iterazione AlphaZero

**Generazioni multiple**: gen2, gen3, gen4...
- Ogni gen: 10k-20k partite di self-play, training rete, validazione
- Tournament periodico: gen N vs gen N-1
- Aspettativa: vediamo trend di crescita su 4-6 generazioni

**Compute serio**: questo è il momento di considerare Lambda Labs se Colab non basta.

### Mesi 5-6 — Validazione finale

- Bot finale vs umani (test reali)
- Tuning finale
- Documentazione

---

## 5. Rischi tecnici reali

Te li elenco onestamente, ognuno con probabilità.

### Rischio 1 — MCTS troppo lento (probabilità 40%)
Risiko ha branching factor enorme. 800 simulazioni per mossa potrebbero richiedere 30+ secondi.
**Mitigazione**: ridurre simulazioni, MCTS parallelizzato, action space ulteriormente ridotto.

### Rischio 2 — Self-play instabile (probabilità 30%)
Le reti potrebbero collassare in strategie banali (rock-paper-scissors fra generazioni).
**Mitigazione**: population-based training (più reti contemporaneamente), regularization.

### Rischio 3 — Compute insufficiente (probabilità 50%)
100€/mese potrebbero non bastare per generazioni serie.
**Mitigazione**: ridurre dimensioni rete, ridurre episodi per generazione, accettare progresso più lento.

### Rischio 4 — Implementazione MCTS buggy (probabilità 20%)
MCTS è notoriamente difficile da debuggare. Bug subtle non si vedono per settimane.
**Mitigazione**: test rigoroso, scenari noti, validazione su giochi semplici prima di Risiko pieno.

### Rischio 5 — Bot evolve strategie "aliene" (probabilità 25%)
Il self-play potrebbe scoprire strategie che battono altri bot ma falliscono contro umani.
**Mitigazione**: validazione regolare contro baseline + giocatori veri (anche tu, in test).

### Rischio 6 — False positive in 1v1 (probabilità 35%)
MCTS in 1v1 può funzionare benissimo (gioco di conquista pura) e poi collassare in 4-player (gioco politico/posizionale). Validare solo in 1v1 sarebbe pericoloso.
**Mitigazione**: a fine Mese 1, oltre a "MCTS vs random in 1v1", testare anche "MCTS vs baseline PPO 29% in 1v1". Solo se MCTS batte ENTRAMBI procediamo a 4-player. Inoltre, alla prima estensione 4-player, test critico: MCTS gen0 4-player vs baseline 29% 4-player. Se collassa qui, è il rischio 6 che si è materializzato.

---

## 6. Cosa NON è incluso

Per evitare scope creep, dichiaro esplicitamente cosa **non** faremo:

- ❌ Stage A2/B/C/D del vecchio piano (PPO-based, abbandonato)
- ❌ Behavior cloning su replay umani (insufficienti dati)
- ❌ Reward shaping basato su 4 regole strategiche (devono emergere da MCTS)
- ❌ UI di gioco / interfaccia visuale (focus su core AI)
- ❌ Multi-mappa (solo mappa standard 42 territori)

---

## 7. Cosa salviamo dal vecchio progetto

- ✅ **Tutto l'env Risiko** (motore di gioco, regole, sdadata, obiettivi). Funziona, è testato. Oro.
- ✅ **Encoding stato 318** come input alla rete
- ✅ **Test suite (148 test)** per validazione regressione
- ✅ **Baseline PPO 29%** come benchmark di confronto
- ✅ **Scripts di valutazione** (valuta_completo.py, batch_eval.py)
- ✅ **bot_rl_opponent.py** per usare modelli come avversari nelle simulazioni MCTS

---

## 8. Setup tecnico

### 8.1 Ambiente
- Python 3.10+
- PyTorch (per rete neurale, lasciamo Stable-Baselines3)
- NumPy, scipy
- Env Risiko esistente

### 8.2 Compute
- **Mese 1-2**: Colab Pro+ (50€/mese). Sviluppo + prototipo.
- **Mese 3+**: rivalutare. Probabilmente Lambda Labs (~70€/mese) per training serio + Colab Pro (10€) per sviluppo.

### 8.3 Repository
Nuovo branch `alphazero-v2` nel repo esistente `EdoMusu1991/risiko-rl`. Non distruggiamo il main (PPO baseline).

### 8.4 Pacchetti nuovi richiesti
- `torch` (rete)
- Implementazione MCTS: scriverò io (no librerie esterne, troppo specifiche)

---

## 9. Modalità di lavoro

### 9.1 Cadenza
- **Settimanale**: review dei progressi, decisione next steps (1-2h)
- **Bi-settimanale**: documento di stato del progetto (cosa fatto, cosa pianificato)

### 9.2 Decisioni
- **Tecniche piccole**: decido io (es: scelta libreria specifica)
- **Tecniche medie**: decidi tu dopo che ti spiego (es: dimensioni rete)
- **Strategiche**: decidi tu (es: cambio paradigma, abbandono progetto)

### 9.3 Validazione
Ogni settimana, test obbligatorio:
- I 148 test esistenti devono restare verdi
- Aggiungiamo test specifici per MCTS (riproducibilità, correttezza)
- Validation periodica: il nuovo bot batte la baseline? Sì/No.

### 9.4 Trasparenza sui risultati negativi
Se a fine mese 1 MCTS non funziona, **te lo dico onestamente** e cambiamo strategia. Niente illusioni.

---

## 10. Prossimi passi immediati

### Step 1 — Tu confermi questo documento
Lo leggi, mi dici cosa non torna o vuoi cambiare.

### Step 2 — Setup branch
- Creo branch `alphazero-v2` nel repo
- Aggiungo cartella `mcts/` per il nuovo codice
- Setup PyTorch sull'ambiente

### Step 3 — Sviluppo settimana 1 (Wrapper MCTS-friendly)
- Modifiche all'env per supportare snapshot/restore
- Test riproducibilità
- Documentazione tempo per mossa

### Step 4 — Validazione baseline
- Confronto: env modificato vs env originale, comportamento identico
- Compute time delle modifiche

---

## 11. Check finale prima di partire

Le 4 cose che voglio sentirti confermare prima di scrivere codice:

1. **Hai letto questo documento?** Sì/No
2. **C'è qualcosa che non torna o vuoi cambiare?** (rispondi specifico)
3. **Sei consapevole del rischio (40% di insuccesso)?** Sì/No
4. **Confermi 6 mesi di impegno e 100€/mese di compute?** Sì/No

Quando hai 4 sì, partiamo.

---

**Fine specifica v2.**
