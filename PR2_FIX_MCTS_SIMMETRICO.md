# PR2 — Fix MCTS simmetrico

PR2 estende il pattern dei due env templati (introdotto da PR1) anche dentro il rollout MCTS, risolvendo:
1. Il crash `assert step() chiamato fuori turno bot. Corrente: ROSSO` che esplodeva durante MCTS
2. Il bug di segno latente sull'observation orientata su `bot_color` quando MCTS espandeva nodi del giocatore avversario

## File modificati / aggiunti

- `alphazero/selfplay/simulate.py` — modificato (aggiunta `simulate_simmetrico`, originale invariato)
- `alphazero/selfplay/search.py` — modificato (aggiunta `search_simmetrico`, originale invariato)
- `alphazero/selfplay/self_play.py` — modificato (`gioca_partita_selfplay_simmetrica` ora usa `search_simmetrico`, firma esterna invariata)
- `alphazero/selfplay/__init__.py` — modificato (esporta le nuove funzioni)
- `tests/test_simulate_simmetrico.py` — nuovo

## Cosa fa il fix

`simulate_simmetrico(root, envs, net, ...)` accetta un dict `envs={"BLU": env_blu, "ROSSO": env_rosso}` invece del singolo env. Internamente:

1. **Restore con riallineamento**: `_restore_e_riallinea` restora `envs[node.player_to_move]` con `node.snapshot`, poi alias `stato`/`rng` sull'altro env. Cosi' i due env vedono lo stesso stato fisico.

2. **Switch al cambio turno**: dopo ogni `env.step()`, se `sotto_fase=None` e partita non terminata, `_switch_se_turno_finito` cambia `env_attivo` all'altro colore + alias + `_inizia_fase_tris()`.

3. **Snapshot post-switch (Opzione Y)**: i nodi figli vengono snapshottati DOPO l'eventuale switch. Questo garantisce l'invariante chiave: `node.snapshot` e' sempre compatibile con `envs[node.player_to_move]`.

`search_simmetrico` e' un wrapper banale di `simulate_simmetrico`. La funzione `gioca_partita_selfplay_simmetrica` di PR1 e' stata modificata internamente per usare `search_simmetrico` (firma esterna identica).

## Invariante critica

```
Per ogni Node n non terminale:
  n.snapshot e' stato preso da envs[n.player_to_move]
  dopo aver eseguito _inizia_fase_tris() su quell'env
```

Conseguenza: al restore di un nodo, restoriamo su `envs[n.player_to_move]`, riallineiamo l'altro env (alias stato/rng), e siamo "pronti a giocare" senza magie extra. Tutti gli accessi successivi (`_costruisci_observation`, `get_action_mask`, `step`) usano `env_attivo.bot_color = n.player_to_move`, quindi sono automaticamente orientati sul giocatore giusto.

## Bug risolto: observation orientation

In `simulate` originale, quando MCTS espande un nodo dove `player_to_move=ROSSO` mentre l'env ha `bot_color=BLU`, `_costruisci_observation()` restituisce un'observation BLU-centric. La rete ritorna un value che la rete ha imparato a riferire a BLU, ma `simulate.py` lo passa a `backup` con `leaf_player=ROSSO`. Quindi:
- value e' "valore per BLU"
- backup lo interpreta come "valore per ROSSO"
- Bug di segno

In `simulate_simmetrico`, l'observation viene costruita da `env_attivo` dove `env_attivo.bot_color == node.player_to_move` per invariante. Quindi value e' nativamente dal POV giusto. Bug eliminato.

## Risultati test

`tests/test_simulate_simmetrico.py` — **7/7 PASSED**:

```
[test_1_simulate_non_crasha]                              -> OK: root.N=1
[test_2_root_ha_figli_dopo_simulazioni]                   -> OK: 1 figli, N=5
[test_3a_simulate_da_root_rosso]                          -> OK: 10 sim da root ROSSO
[test_3b_albero_misto_quando_fine_turno_e_vicino]         -> OK: albero ha BLU+ROSSO
[test_4_selfplay_mcts_genera_sample_da_entrambi_i_colori] -> OK: BLU=105, ROSSO=122
[test_5_no_assert_step_fuori_turno]                       -> OK: 30 sim senza assert
[test_6_env_restorato_dopo_simulate]                      -> OK: regola d'oro rispettata
```

Highlight di test_3b (il check più rilevante della logica MCTS): partendo da una root in fase ATTACCO (vicino al fine turno), 50 simulazioni espandono 182 nodi che contengono **sia BLU sia ROSSO**. Quindi MCTS sta veramente espandendo nodi avversario, non delegando a bot interno.

Highlight di test_4 (validazione end-to-end): partita completa con MCTS+rete vera (n_simulations=5, max_decisioni=300) genera **105 sample BLU + 122 ROSSO**. Niente assert, niente crash.

## Test di non regressione

- `tests/test_selfplay_simmetrico.py` (PR1) — **6/6 PASSED**
- `tests/test_alphazero_simulate.py` (legacy) — **9/9 PASSED** (usa `simulate` originale)
- `tests/test_alphazero_search_selfplay.py` (legacy) — **9/9 PASSED** (usa `search`/`gioca_partita_selfplay` originali)

I file legacy non sono stati toccati: aggiungiamo accanto invece di sostituire.

## Decisioni di design prese

1. **Aggiunta accanto, non sostituzione**. `simulate`, `search` e `gioca_partita_selfplay` legacy restano intatti. `simulate_simmetrico`, `search_simmetrico` e `gioca_partita_selfplay_simmetrica` sono le versioni AlphaZero-pure. Backward-compat totale, rollback istantaneo.

2. **Opzione Y (snapshot post-switch)**: i nodi figli vengono snapshottati DOPO il switch + `_inizia_fase_tris`. Cosi' al restore basta `envs[n.player_to_move].restore(n.snapshot)` senza condizioni speciali.

3. **Niente refactoring di env.py**: sarebbe stato il fix "alla radice" (rendere bot_color dinamico), ma con superficie di test enorme. Strada B (due env templati) ha effetto equivalente con cambio chirurgico.

4. **Limite a 1v1**: lo stesso vincolo di PR1 (`mode_1v1=True`). 4-player richiederebbe 4 env e una decisione su come gestire VERDE/GIALLO (rete unica? bot interno?). Non in scope.

## Casi particolari gestiti

- **Step terminale (partita finisce dentro MCTS)**: env_attivo non switcha (`stato.terminata=True` blocca lo switch). Il child terminale prende `player_to_move = stato.giocatore_corrente` (che non cambia in `termina_partita_per_*`). `terminal_value` = reward dell'env attivo, dal POV di parent.player_to_move = child.player_to_move (verificato con assert).

- **Action forzata (n_legali=1)**: nessun MCTS, nessun sample, solo step. Inalterato rispetto a PR1.

- **Truncated (max_decisioni)**: simulate non viene chiamata oltre il limite. Lo stato finale e' "no winner" e value_target=0 per entrambi i colori (gia' gestito da PR1).

## Prossimi passi suggeriti

1. **Mini training validation**: lanciare un training breve (es. 20 partite con n_simulations=20, 1 epoch su batch piccolo) per verificare che la loss scenda. PR2 e' validato a livello di flow, ma il "vero" test e' che la rete impari.

2. **Performance check**: con due env e MCTS simmetrico, il tempo per partita e' raddoppiato rispetto a PR0/PR1 (ogni step MCTS deve gestire eventualmente switch + alias). Misurare se l'overhead e' accettabile prima di lanciare training lunghi.

3. **Eventuale PR3 (parallel self-play)**: ora che il flow e' AlphaZero-puro, parallelizzare le partite (multiprocessing) e' la cosa che da' la velocita' vera. Era nelle note di sessione.
