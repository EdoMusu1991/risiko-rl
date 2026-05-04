# Diario Settimana 3 — Audit input + Rete neurale

**Stato**: Completato ✅
**Data**: maggio 2026

## Obiettivi della settimana

> Inizio AlphaZero v2: audit normalizzazione input, design rete, implementazione, test
> NB: tutto validato da ChatGPT prima di scrivere codice

## Cosa è stato fatto

### Step 1 — Audit normalizzazione input ✅

Lanciato script di analisi su 1064 osservazioni reali (Stage A2 attivo):

**Risultato eccellente**:
- **Tutte le 342 feature sono già in [0, 1]**
- Range globale: min=0.000, max=1.000
- **Niente normalizzazione necessaria**

Note:
- L'observation in 1v1 ha 137 feature costanti (~40%) — sono gli slot per VERDE/GIALLO morti. Non è bug, è strutturale.
- In 4-player solo 33 feature costanti (~10%, normale).

ChatGPT aveva detto "priorità 1 normalizzazione". L'avevamo già fatta in fase di encoding (Stage A2).

### Step 2 — Design rete validato da ChatGPT ✅

Modifiche dal mio design iniziale:
- Tronco: **256 → 256 → 128** (non 256-256-256, ChatGPT)
- Action mask: usare **-1e9** non -inf (ChatGPT — evita NaN)
- Inizializzazione **Xavier** sulle teste finali (ChatGPT)
- **PUCT, non UCB** (ChatGPT — critico per usare prior della rete)
- **Backup value con cambio segno** in giochi a turni alterni (ChatGPT — bug più comune)

### Step 3 — Implementazione rete ✅

File: `alphazero/network/model.py` (~180 righe)

Architettura:
```
Input [342]
  ↓
Tronco condiviso: Dense(256) → Dense(256) → Dense(128) + ReLU
  ↓
  ┌──────────────────┴──────────────────┐
  ↓                                     ↓
Policy head:                        Value head:
  Dense(256) → ReLU                   Dense(64) → ReLU
  Dense(1765) [logits]                Dense(1) → tanh
```

Parametri totali: **681.446** (~600k come previsto).

Componenti:
- `RisikoNet`: classe principale
- `apply_mask_and_softmax`: helper per mascherare azioni illegali
- `alphazero_loss`: MSE(value) + CrossEntropy(policy) standard

### Step 4 — Test e validazione ✅

File: `tests/test_alphazero_net.py` (9 test)

1. ✓ Creazione rete con ~600k parametri
2. ✓ Forward single sample
3. ✓ Forward batch (32)
4. ✓ Mask + softmax (prob illegali = 0, legali = 1)
5. ✓ Loss AlphaZero produce valore finito
6. ✓ Gradient flow su 14/14 parametri
7. ✓ Rete reagisce a input diversi
8. ✓ Integrazione con observation reale di RisikoEnv
9. ✓ La rete impara un pattern lineare (v_loss da 0.047 a 0.026)

**174 test totali verdi** (165 originali + 9 nuovi).

## Stato repo

Nuove cartelle:
```
alphazero/
├── __init__.py
├── network/
│   ├── __init__.py
│   └── model.py     ← rete neurale RisikoNet
├── training/        (vuoto — Settimana 5)
├── selfplay/        (vuoto — Settimana 4)
└── eval/            (vuoto — Settimana 6)
```

## Prossimi step (Settimana 4)

1. **MCTS guidato dalla rete**: modifica `mcts/mcts_base.py` esistente per usare prior della rete invece di UCB1 puro
2. **Implementazione PUCT**: formula Q + c * P * sqrt(N_parent) / (1 + N_child)
3. **Backup con cambio segno**: gestire correttamente in azione space gerarchico Risiko
4. **Test integrazione**: 1 partita end-to-end con rete random
5. **Aspettativa onesta** (timeline ChatGPT): 
   - Settimana 1 (= 4): rete random, gioca male
   - Settimana 2 (= 5): inizia a capire dopo training
   - Settimana 3-4 (= 6-7): supera euristico
   - Mese 2: supera baseline PPO 29%

## Lezioni della Settimana 3

1. **Audit prima di codice**: la priorità "normalizzazione" si è rivelata un non-problema. Bene aver verificato.

2. **Design validato prima di codice**: ChatGPT ha trovato 3 errori critici nel mio design iniziale. Senza la sua revisione avrei perso giorni.

3. **Test rete IMPARA il pattern**: critico aver fatto il test "value = funzione lineare delle features". È il modo per essere sicuri che l'architettura non abbia bug strutturali. Senza, scoprivi i bug solo a Settimana 6 con training reale.

4. **Patience**: non lanciato ancora training reale. Settimana 4 = MCTS+rete, Settimana 5 = self-play loop, Settimana 6 = primi risultati. Niente fretta.

## Cosa NON ho fatto

- Non ho integrato il bot Java (rinviato a Mese 2-3 come da raccomandazione ChatGPT)
- Non ho lanciato training (verrà in Settimana 5-6)
- Non ho ottimizzato l'action space 1765 (rinviato — ChatGPT: "non ottimizzare prima di vedere se serve")
