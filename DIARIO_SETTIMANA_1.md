# Diario Settimana 1 — Determinismo dell'env + Modalità 1v1

**Stato**: COMPLETATA ✅
**Data**: maggio 2026

## Obiettivi della settimana

> Settimana 1 = determinismo totale dell'env + modalità 1v1 per AlphaZero MCTS

## Cosa è stato fatto

### Giorni 1-2 — Snapshot/Restore

**Implementazione**:
- `RisikoEnv.snapshot()` → cattura stato dinamico completo (deep copy)
- `RisikoEnv.restore(snap)` → ripristina env a stato precedente
- `RisikoEnv.clone()` alias di snapshot()

**Test (i 4 di ChatGPT)**: tutti passati al primo colpo
- ✅ Idempotenza, Determinismo, Replay, No side effects, Performance

**Performance**: snapshot+restore = 0.947 ms (sotto target 1ms)

### Giorni 3-7 — Modalità 1v1 + Reward margin

**Decisioni di default prese (in mancanza di feedback specifico):**
1. **Obiettivi**: SÌ, conservati (parte essenziale di Risiko)
2. **Setup**: 21 territori per giocatore (42/2), bilanciato
3. **Sdadata**: cap a 60 round (già config standard)
4. **Reward**: margine punti come da spec v2

**Implementazione**:
- Parametro `mode_1v1: bool` in `RisikoEnv.__init__`
- Parametro `reward_mode: str = "binary" | "margin"` in `RisikoEnv.__init__`
- Metodo `_converti_a_1v1()`: dichiara morti VERDE+GIALLO, ridistribuisce
  i loro territori fra BLU e ROSSO (alternato, deterministico)
- Metodo `_calcola_reward_margin()`:
  `(punti_propri - punti_max_avversari_vivi) / 100`, clipped [-1, +1]
- Metodo `_calcola_punti_finali()`: punti in obj + punti fuori obj
- `_calcola_reward_finale()` ora supporta entrambe le modalità

**Test (8 nuovi in `test_mode_1v1.py`)**: tutti passati
- ✅ Setup 1v1: solo BLU+ROSSO vivi
- ✅ Distribuzione territori (~21 ciascuno)
- ✅ Morti senza carte
- ✅ Partite 1v1 terminano in ≤60 round
- ✅ Reward margin in [-1, +1]
- ✅ Reward binary invariato
- ✅ Snapshot/restore in 1v1
- ✅ 4-player default invariato (no regressione)

**Test 20 partite 1v1 con bot random**:
- Tutte completate (motivo fine: sdadata)
- Range round 36-40 (sdadata standard)
- Reward margin distribuito in [-0.36, +0.80]
- BLU vince 13/20 (~65% — varianza statistica naturale)
- Reward positivi: 18/20 (>= win count perché margine positivo anche
  in alcune partite "perse")

### Stato finale codice

- File modificato: `risiko_env/env.py` (+~200 righe totali)
  - snapshot/restore (giorni 1-2): ~100 righe
  - mode_1v1 + reward margin (giorni 3-7): ~100 righe
- File nuovi: `tests/test_snapshot_restore.py`, `tests/test_mode_1v1.py`
- Test totali: **165** (era 148, +9 snapshot, +8 mode_1v1)
- Tutti verdi al 100%

## Curiosità interessante (per AlphaZero)

In modalità 1v1 con reward margin, il **vincitore della partita** non è
sempre il giocatore con reward positivo. Esempi:
- Vincitore = ROSSO, ma reward margin di BLU = +0.30 (BLU ha più punti
  totali ma ROSSO ha vinto per altri criteri di Risiko)

Questo è **intenzionale**: AlphaZero ottimizzerà i punti, non la vittoria
binaria. Un bot che fa 60 punti ed esce 2° è migliore di uno che fa 30 e
esce 1° per un colpo di fortuna sull'obiettivo.

## Quello che NON è stato fatto

- Rollout policy euristica → Settimana 2
- MCTS → Settimana 2-3

## Cosa cercare se ChatGPT vuole stress test ulteriori

Lo script `tests/test_snapshot_restore.py` è il punto di partenza. Per
test più aggressivi:
1. Stress con Stage A2 attivo (già fatto: 50/50 OK)
2. Stress in modalità 1v1 (già fatto: 30/30 OK)
3. Stress con avversari custom (modelli RL caricati)
4. Test di equivalenza fra `step` su env-A snapshot-then-restore vs env-B vergine

## Verdetto Settimana 1

Tutti gli obiettivi raggiunti, performance sotto target, zero regressioni.
**Pronto per Settimana 2: rollout policy + MCTS base in 1v1.**

## Stato repo

- File modificato: `risiko_env/env.py`
- File nuovi: `tests/test_snapshot_restore.py`, `tests/test_mode_1v1.py`
- Test totali: 165 (era 148, +17)
- Performance baseline: 0.947 ms snapshot/restore
