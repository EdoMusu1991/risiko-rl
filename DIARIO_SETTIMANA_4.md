# Diario Settimana 4 — MCTS guidato dalla rete

**Stato**: Completato ✅
**Data**: maggio 2026

## Obiettivi della settimana

> Settimana 4: MCTS+rete, Node, PUCT, backup, simulate, search, self-play
> Target ChatGPT: "far funzionare UNA singola ricerca MCTS corretta"

## Sub-step completati

### Sub-step 1 — Node class ✅
File: `alphazero/selfplay/node.py`

- 12 attributi (N, W, P, children, parent, action_taken, snapshot, legal_actions, is_terminal, terminal_value, player_to_move)
- `__slots__` per memoria
- `Q` come property
- 6 test verdi

### Sub-step 2 — PUCT selection ✅
File: `alphazero/selfplay/selection.py`

- `select_child(node, c_puct=1.5)`: formula PUCT standard AlphaZero
- `select_action_from_root(root, temperature)`: con T=0 (argmax) o T=1 (sampling)
- 8 test verdi

### Sub-step 3 — backup con cambio segno ✅
File: `alphazero/selfplay/backup.py`

- `backup(path, value, player_leaf)`: confronta ogni nodo con player_leaf (NON con last_player che cambia)
- Importante per Risiko (sequenze BLU×3 → ROSSO×3 non alternate)
- **14 test verdi inclusi i 4 "perfetti" di ChatGPT**

### Sub-step 4 — simulate() ✅
File: `alphazero/selfplay/simulate.py`

- Selection → Expansion (con rete) → Backup
- `try/finally` per ripristinare env (regola d'oro ChatGPT)
- Espande TUTTI i figli legali in una volta (come scheletro ChatGPT)
- **9 test verdi inclusi i 6 "perfetti" di ChatGPT**

### Sub-step 5 — search() + self-play ✅
File: `alphazero/selfplay/search.py`, `alphazero/selfplay/self_play.py`

- `search(root, env, net, n_simulations, temperature)`: loop di simulate + estrazione
- `visite_to_policy_full(root, action_dim)`: vettore 1765-D per il replay buffer
- `gioca_partita_selfplay()`: orchestra una partita, raccoglie TrainingSample
- 9 test verdi

## Statistiche

- **220 test verdi totali** (su 16 file di test)
- **+45 test nuovi** rispetto a Settimana 3
- Codice nuovo: ~700 righe in `alphazero/selfplay/`

## Performance reali (CPU)

- search 50 sim su fase rinforzo: ~2.2s (22 sim/s)
- Partita self-play short (n_sim=5, max=20): ~3s
- Partita completa (n_sim=10): timeout >120s su CPU

**Su GPU atteso 5-10x speedup** — provare in Colab Pro+ in Settimana 5.

## Validazioni di ChatGPT

ChatGPT ha fornito **due batterie di test critici** che hanno guidato la validazione:

1. **4 test "perfetti" per backup**: sequenza BLU×3→ROSSO×2, value negativo, accumulo, ordine path
2. **6 test "perfetti" per simulate**: env non sporcato, root espansa, figli legali, prior somma 1, child snapshot ripristinabile, seconda simulate scende

**Tutti passati al primo colpo, nessuna modifica al codice.**

Questo è significativo perché ChatGPT ha avvisato:
> "simulate() e' dove si rompe davvero tutto (90% dei progetti)"

Il fatto che siano passati subito significa che il design (suo) era corretto e l'implementazione fedele.

## Cosa NON è ancora fatto (Settimana 5)

- ❌ Replay buffer
- ❌ Training loop (loss + optimizer + checkpoint)
- ❌ Pipeline self-play distribuita
- ❌ Warm-start con rollout euristico (opzionale)
- ❌ Dirichlet noise (opzionale)
- ❌ Eval pipeline (gen N vs gen N-1, vs baseline 29% PPO)

## Prossimi step (Settimana 5)

1. Replay buffer (deque con maxlen=100k)
2. Training loop minimale: sample batch → forward → loss → backward
3. Test che la rete impara su batch di samples reali
4. Eseguire in Colab Pro+ con GPU
5. Prima generazione vs random (n_partite=20)

## Lezioni apprese

1. **Tutti i test ChatGPT al primo colpo**: il design "espandi tutti i figli in expansion + backup con player_leaf fisso" era corretto. Niente ore di debugging.

2. **CPU è lento**: 220 test passano ma una partita reale (250+ dec MCTS) timeout. Indispensabile passare a GPU in Settimana 5.

3. **Risiko non e' scacchi**: action space gerarchico, sequenze stesso giocatore, partite lunghe. Il codice generico AlphaZero funziona, ma serve attenzione (es. fast-path per 1 azione legale).

4. **Self-play in 1v1 simmetrico**: stessa rete, dataset doppio. La simmetria value (BLU vs ROSSO) e' un ottimo invariante per debug.

## Cosa portare avanti

- L'observation 342-D normalizzata in [0,1] è un asset solido
- snapshot/restore env veloce (1ms) abilita MCTS pratico
- Bot euristico Settimana 2 (91% vs random) può servire come warm-start in Settimana 5
- Baseline PPO 29% va salvata e usata per benchmark a fine Mese 1 v2
