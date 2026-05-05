# PR1 — Fix self-play simmetrico

Riepilogo delle modifiche, decisioni di design e risultati.

## File modificati / aggiunti

- `alphazero/selfplay/self_play.py` — modificato (aggiunta funzione, docstring di testa aggiornato)
- `tests/test_selfplay_simmetrico.py` — nuovo

La funzione legacy `gioca_partita_selfplay` è **intatta**, per A/B test e rollback istantaneo.

## Cosa fa il fix

`gioca_partita_selfplay_simmetrica` usa **due env templati** (`env_blu` con `bot_color=BLU`, `env_rosso` con `bot_color=ROSSO`), entrambi con `_skip_giro_avversari=True`. Stato e rng sono condivisi via re-aliasing al cambio turno:

```
env_attivo.step(...) → step finale: sotto_fase=None
                      → _fine_turno_bot interno (pesca + sdadata + avanza_turno)
                      → exit step
env_passivo.stato = env_attivo.stato     # re-aliasing critico
env_passivo.rng   = env_attivo.rng
env_passivo._inizia_fase_tris()
env_attivo = env_passivo                  # swap
```

Il pattern è la "promozione" al main loop di quello che già fa `bot_rl_opponent.gioca_turno_rl` (mini-env condiviso). Niente refactoring di `env.py`, niente nuovo modulo.

## Decisioni di design prese

1. **`_skip_fine_turno_bot` resta `False` su entrambi gli env**. Lasciamo che sia `step()` di env_attivo a fare pesca+sdadata+avanza, perché usa `self.bot_color` che coincide col giocatore corrente. Niente gestione esplicita nel loop.

2. **Re-aliasing critico**: `env.restore(snap)` riassegna `self.stato = copy.deepcopy(...)` e ricrea `self.rng`. Quindi dopo ogni MCTS search, `env_attivo.stato` è un oggetto **nuovo**. Riallineare env_passivo solo al cambio turno è semplice e sicuro.

3. **value_target ANTISIMMETRICO ±1** (AlphaZero-style), NON `_calcola_reward_finale`. Spiegazione: `_calcola_reward_finale` usa `REWARD_PER_POSIZIONE` che dà +0.3/-0.3 ai posti intermedi. In 1v1 con vincitore=BLU darebbe `reward_blu=+1.0` e `reward_rosso=+0.3` (entrambi positivi!). Pessimo per la value head: imparerebbe che "stare lì a fine partita è buono per chiunque". Usato invece `vincitore == player_at_state ? +1 : -1`, con `0` per pareggio (truncated o cap_sicurezza con punteggi pari).

   Nota: la legacy `gioca_partita_selfplay` "funzionava per caso" perché in 1v1 con un solo bot_color non vedeva mai entrambi i lati dell'asimmetria.

4. **Limite a `mode_1v1=True`** (assert). Il caso 4-player richiede design separato (4 env? rete che gioca solo BLU/ROSSO con VERDE/GIALLO via bot interno?). Non in scope per PR1.

5. **Parametro `policy_fn` opzionale** per sostituire MCTS+rete nei test. Default `None` = comportamento normale (MCTS+rete). Permette test smoke senza torch/network real.

## Bug trovato durante l'implementazione

Prima versione del codice usava `value_target_altro = -reward_finale`, assumendo anti-simmetria di `_calcola_reward_finale`. **Sbagliato**: il primo run ha mostrato vincitore=BLU con `v_blu=-0.3` e `v_rosso=+0.3`. Il test "segni opposti" passava lo stesso, perché era troppo debole (è anti-simmetrico anche +0.3/-0.3 ma con segno invertito rispetto al vincitore).

Test rinforzato a `test_value_target_vincitore_perdente` che verifica esplicitamente:
- vincitore=BLU → v_blu=+1.0, v_rosso=-1.0
- vincitore=ROSSO → v_blu=-1.0, v_rosso=+1.0
- pareggio → entrambi 0.0

## Risultati test

```
ALL 6 TESTS PASSED:
  test_partita_simmetrica_genera_sample_da_entrambi_i_colori
  test_value_target_vincitore_perdente
  test_seed_riproducibile
  test_seed_diversi_partite_diverse
  test_struttura_sample
  test_verbose_non_crasha
```

Stress test su 20 seed (random play, no MCTS):
- Simmetria sample: ovunque BLU > 0 E ROSSO > 0 (es. seed 4: 540/540, seed 8: 645/656)
- Distribuzione vincitori: 15 BLU, 4 ROSSO, 1 pareggio (truncato a 2000 dec). Bias BLU atteso (muove primo → vantaggio carte/conquiste in random play).
- Motivi fine variabili: sdadata (16), obiettivo_completato (3), None/truncated (1). Tutti gestiti.
- value_target coerente in tutti i 20 casi.

## Come usare la funzione (production, con MCTS+rete)

Drop-in replacement della legacy, ma **non prende l'env in input** — lo crea internamente:

```python
from alphazero.selfplay.self_play import gioca_partita_selfplay_simmetrica

samples, stats = gioca_partita_selfplay_simmetrica(
    net=la_mia_rete,
    n_simulations=50,
    seed=42,
    mode_1v1=True,        # obbligatorio (assert)
)

# Adesso:
#   stats["n_samples_blu"]   > 0
#   stats["n_samples_rosso"] > 0
```

## Cosa NON fa questo PR (rimane in PR2)

1. **MCTS interno simmetrico**: `simulate.py` chiama ancora `env.step()` con `_skip_giro_avversari=False` (default), quindi durante un rollout MCTS i turni avversari vengono giocati con bot interno random/euristico. MCTS non sta espandendo veramente i nodi avversario.

2. **Observation orientation durante MCTS**: quando `simulate()` valuta un nodo dove `player_to_move != env.bot_color`, l'observation è ancora "bot_color-centric" (BLU-centric in env_blu). Il value head ritorna un valore che la rete ha imparato a riferire al bot_color, ma `Node.player_to_move` lo interpreta dal punto di vista del giocatore corrente. Bug di segno latente.

   **Nota positiva**: nei sample raccolti da `gioca_partita_selfplay_simmetrica`, l'observation è già "player-centric" perché viene da env_attivo, che è allineato sul giocatore corrente. Quindi la rete addestrata da PR1 imparerà già la giusta orientazione per gli stati visti dal main loop. Il bug è solo durante MCTS expansion.

3. **Refactoring di `_calcola_reward_finale`** per essere antisimmetrico in 1v1. Non l'ho toccato — PR1 ha semplicemente bypassato quella funzione per il calcolo del value_target. Se in futuro si vuole eliminare l'incoerenza alla radice, è un cambio separato.

## Suggerimento per la prossima sessione

Prima di procedere a PR2 (che è più delicato), validare PR1 con un training breve:
1. Un round di self-play simmetrico con MCTS+rete reale (n_simulations=20, n_partite=50).
2. Verificare che `n_samples_blu` ≈ `n_samples_rosso` (entro ±15%) anche con MCTS.
3. Verificare distribuzione vincitori non degenere (non 100% BLU).
4. Un mini-training step e check che la loss scenda.

Se questo va, PR2 può partire con sicurezza.
