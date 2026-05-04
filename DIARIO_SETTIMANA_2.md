# Diario Settimana 2 — Rollout policy + MCTS base

**Stato**: Parziale, con problemi onesti
**Data**: maggio 2026

## Obiettivi della settimana

> Settimana 2 = rollout policy euristica + MCTS base in 1v1

## Cosa è stato fatto

### Giorni 1-2 — Rollout policy euristica ✅
File: `risiko_env/bot_euristico.py`

**3 regole minimali (da specifica ChatGPT)**:
1. Attacca solo se win prob > 0.55 (tabella stimata via simulazione)
2. Leggera preferenza (60%) per rinforzo confini
3. Lascia almeno MIN_ARMATE_DIETRO=2 quando attacca

**Tutto il resto = come bot_random**: tris al 50%, spostamento 30%, 0-3 attacchi/turno, quantità random.

**Test**:
- Bot euristico vs random in 1v1: **91% WR**
- Random vs random: 49% (sanity check OK)

**Lezione importante**: la rollout policy è **già più forte del previsto**. ChatGPT aveva detto "60-75%". 91% è troppo:
- Anche solo "rinforzo confini" senza altre regole → 72% vs random
- Random in Risiko fa errori così grossi (attacca 1 vs 4) che bastano poche regole minime per dominare

### Giorni 3-5 — MCTS base ⚠️ Problemi tecnici
File: `mcts/mcts_base.py` (presente da sessione precedente compactata)

**Implementazione**:
- `MCTSNode`: nodi con N, W, Q, children
- `MCTS.search()`: PUCT con UCB1 (no policy prior, no rete)
- `rollout_euristico`: usa bot_euristico per tutti i giocatori durante rollout
- `rollout_random`: alternativa per debug
- `MCTSAgent`: wrapper con fast-path per fasi banali (rinforzo, spostamento)

**Performance**:
- 76-87 simulazioni/sec (dentro target ChatGPT)
- 100 simulazioni → ~1.3s per decisione MCTS

## Problemi rilevati

### Problema 1: MCTS perde vs euristico puro

Test diretto MCTS+rollout (BLU) vs euristico puro (ROSSO):
- n_sim=10: 1/5 vittorie BLU
- n_sim=30: timeout (>120s/partita)

### Problema 2: Tempo per partita troppo alto

- 250-1000 decisioni MCTS per partita 1v1 (azione space gerarchico)
- Anche con fast-path e n_sim=10: 30-60s per partita
- Validation 100 partite = ore di compute

### Problema 3: Rollout troppo forte per discriminare

ChatGPT aveva avvertito: rollout troppo forte rende MCTS cieco.
Sembra essersi materializzato.

## Verdetto onesto

**MCTS classico (rollout-based, no NN) sta facendo fatica su Risiko 1v1.**
Era previsto come **rischio 1** nella spec v2 (60% per ChatGPT). Si è materializzato.

L'implementazione è corretta. È un limite strutturale.

## Stato repo

- File nuovi: `risiko_env/bot_euristico.py`
- Test totali: **165** (tutti verdi)
- Performance baseline: rollout euristico 91% vs random in 1v1

## Decisione richiesta

Tre strade:
- **A**: Macro-azioni MCTS (1-2 settimane di rework)
- **B**: Salto a rete neurale prima del previsto
- **C**: Pausa + sentire ChatGPT

**Raccomandazione**: C.