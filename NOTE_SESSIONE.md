# Note sessione Settimana 6 (in corso)

## Cosa è stato fatto

1. **Smoke test su GPU completato**:
   - GPU funziona (rete forward + training)
   - Test overfit su GPU: loss -96.4% (perfetto)

2. **Fix bug 1 — device transfer in simulate.py**:
   - Errore: tensori MCTS su CPU mentre rete su GPU
   - Fix applicato: `simulate.py` ora sposta tensori su `net.parameters().device`
   - Validato con 9 test verdi

3. **Performance reale misurata su GPU**:
   - Partita short n_sim=20 max=80: 37.8s, 473ms/decisione
   - Partita reale n_sim=50: 11.3 min, 1339ms/decisione
   - **GPU non aiuta** (batch=1 non sfrutta hardware)

4. **Bug 2 scoperto — self-play asimmetrico**:
   - `gioca_partita_selfplay` produce 505 sample BLU, 0 ROSSO
   - Causa: env salta automaticamente i turni di ROSSO con bot interno
   - **Self-play attuale non è AlphaZero puro**, è MCTS+rete (BLU) vs bot interno (ROSSO)

## Cosa NON è stato fatto

- Fix bug 2 (simmetria self-play): tentato ma codice non testato, rollback a versione originale
- Training reale
- Partite parallele

## Da fare in nuova sessione

### Studio preliminare necessario
1. `risiko_env/env.py` — funzioni `_avanza_fino_a_turno_bot`, `_fine_turno_bot`, `_inizia_fase_tris`
2. `risiko_env/sdadata.py` — `gestisci_fine_turno`
3. `risiko_env/bot_rl_opponent.py` — pattern di mini-env condiviso

### Design fix
- Due env separati con `bot_color` diverso, stato condiviso
- Quando turno passa al "non-master", usare mini-env per quel giocatore
- Salvare sample da entrambi
- Validare con test: BLU > 0, ROSSO > 0

### Ordine esecuzione
1. Studio codice env (1h)
2. Design su carta (validare con ChatGPT)
3. Scrittura codice + test
4. Validazione: 1 partita test → samples BLU > 0 e ROSSO > 0
5. SOLO ALLORA training reale

## Decisioni prese da ChatGPT in sessione

- Strategia 2: fix simmetria PRIMA di velocità
- Non fare batched MCTS adesso, prima self-play parallelo se serve
- Non lanciare training finché bug simmetria non risolto

