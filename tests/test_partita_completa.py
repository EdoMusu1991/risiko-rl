"""
test_partita_completa.py — Test di integrazione end-to-end.

Esegue partite complete usando bot random per ogni decisione.
Non testa intelligenza, ma verifica che TUTTI i moduli si parlino
correttamente e che ogni partita termini in modo legale.

Questo è il "smoke test" più importante: cattura bug di interazione
tra moduli che i test unitari non vedono.

Esegui: python tests\test_partita_completa.py
"""

import sys
import os
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env.data import (
    OBIETTIVI,
    COLORI_GIOCATORI,
    ADIACENZE,
    ROUND_CAP_SICUREZZA,
)
from risiko_env.setup import crea_partita_iniziale
from risiko_env.motore import (
    calcola_rinforzi_base,
    calcola_bonus_continenti,
    seleziona_due_tris_disgiunti,
    calcola_bonus_tris,
    gioca_tris,
    piazza_rinforzi,
    attacco_legale,
    territori_attaccabili_da,
    esegui_attacco,
    applica_conquista,
    spostamento_legale,
    esegui_spostamento,
    pesca_carta,
    avanza_turno,
)
from risiko_env.sdadata import gestisci_fine_turno
from risiko_env.obiettivi import determina_vincitore


# ─────────────────────────────────────────────────────────────────────────
#  BOT RANDOM (per smoke test)
# ─────────────────────────────────────────────────────────────────────────

def bot_random_gioca_turno(stato, colore, rng):
    """
    Esegue un intero turno per il giocatore `colore` con strategie random:
    - Tris: gioca se può
    - Rinforzi: distribuzione casuale sui propri territori
    - Attacchi: attacca finché ha ratio favorevole O 5 attacchi max
    - Spostamento: random (50% di probabilità di farlo)
    - Pesca: automatica se ha conquistato
    """
    # ── FASE 1: tris e rinforzi ──────────────────────────────────
    giocatore = stato.giocatori[colore]

    # Gioca tris se possibile
    tris_da_giocare = seleziona_due_tris_disgiunti(giocatore.carte)
    bonus_tris = calcola_bonus_tris(stato, colore, tris_da_giocare) if tris_da_giocare else 0
    if tris_da_giocare:
        gioca_tris(stato, colore, tris_da_giocare)

    # Calcola rinforzi totali
    rinf_base = calcola_rinforzi_base(stato, colore)
    bonus_cont = calcola_bonus_continenti(stato, colore)
    totale_rinforzi = rinf_base + bonus_cont + bonus_tris

    # Distribuzione random
    territori_propri = stato.territori_di(colore)
    if not territori_propri:
        return  # Eliminato di fatto

    if totale_rinforzi > 0:
        distribuzione = {}
        for _ in range(totale_rinforzi):
            t = rng.choice(territori_propri)
            distribuzione[t] = distribuzione.get(t, 0) + 1
        piazza_rinforzi(stato, colore, distribuzione)

    # ── FASE 2: attacchi ─────────────────────────────────────────
    n_attacchi = 0
    max_attacchi = 5
    while n_attacchi < max_attacchi:
        # Trova attacchi possibili
        propri_con_armate = [t for t in stato.territori_di(colore)
                             if stato.mappa[t].armate >= 2]
        if not propri_con_armate:
            break

        # Random: scegli un territorio attaccante
        da = rng.choice(propri_con_armate)
        attaccabili = territori_attaccabili_da(stato, da)
        if not attaccabili:
            n_attacchi += 1
            continue

        # Attacca solo se ratio armate >= 1.5 (per non essere troppo stupido)
        verso = rng.choice(attaccabili)
        rapporto = stato.mappa[da].armate / max(1, stato.mappa[verso].armate)
        if rapporto < 1.5:
            n_attacchi += 1
            continue

        # Esegui l'attacco
        esito = esegui_attacco(stato, colore, da, verso, rng,
                              fermati_dopo_lanci=3)
        if esito.conquistato:
            # Sposta minimo da specifica
            minimo = esito.num_dadi_ultimo_lancio
            massimo = stato.mappa[da].armate - 1
            quantita = min(massimo, max(minimo, massimo // 2))
            fine = applica_conquista(stato, colore, da, verso,
                                     quantita, esito, rng)
            if fine:
                return  # Vittoria immediata, esci dal turno
        n_attacchi += 1

        # Stop se la partita è terminata (es. vittoria immediata)
        if stato.terminata:
            return

    # ── FASE 3: spostamento (random, 50%) ────────────────────────
    if rng.random() < 0.5:
        territori_propri = stato.territori_di(colore)
        candidati = []
        for da in territori_propri:
            if stato.mappa[da].armate < 3:
                continue
            for verso in ADIACENZE[da]:
                if stato.mappa[verso].proprietario == colore:
                    candidati.append((da, verso))
        if candidati:
            da, verso = rng.choice(candidati)
            from risiko_env.motore import _minimo_da_lasciare_per_spostamento
            min_da_lasciare = _minimo_da_lasciare_per_spostamento(stato, da, colore)
            massimo = stato.mappa[da].armate - min_da_lasciare
            if massimo >= 1:
                quantita = rng.randint(1, massimo)
                if spostamento_legale(stato, colore, da, verso, quantita):
                    esegui_spostamento(stato, colore, da, verso, quantita)

    # ── FASE 4: pesca carta ──────────────────────────────────────
    pesca_carta(stato, colore, rng)


# ─────────────────────────────────────────────────────────────────────────
#  SIMULAZIONE PARTITA COMPLETA
# ─────────────────────────────────────────────────────────────────────────

def simula_partita(seed: int, log: bool = False) -> dict:
    """
    Simula una partita completa con bot random.

    Restituisce:
        {
            "vincitore": colore,
            "motivo_fine": stringa,
            "round_finale": int,
            "turni_giocati": int,
            "giocatori_eliminati": list,
            "durata_secondi": float,
        }
    """
    inizio = time.time()
    stato = crea_partita_iniziale(seed=seed)
    rng = random.Random(seed * 7 + 1)  # rng diverso da quello del setup

    turni_giocati = 0
    max_turni_safety = 1000  # Failsafe contro loop infinito

    while not stato.terminata and turni_giocati < max_turni_safety:
        colore = stato.giocatore_corrente
        if colore is None or not stato.giocatori[colore].vivo:
            avanza_turno(stato)
            continue

        # Esegui il turno completo
        bot_random_gioca_turno(stato, colore, rng)
        turni_giocati += 1

        if stato.terminata:
            break

        # Gestisci fine turno (sdadata, cap)
        gestisci_fine_turno(stato, colore, rng)
        if stato.terminata:
            break

        avanza_turno(stato)

    fine = time.time()

    eliminati = [c for c, g in stato.giocatori.items() if not g.vivo]
    return {
        "vincitore": stato.vincitore,
        "motivo_fine": stato.motivo_fine,
        "round_finale": stato.round_corrente,
        "turni_giocati": turni_giocati,
        "giocatori_eliminati": eliminati,
        "durata_secondi": fine - inizio,
    }


# ─────────────────────────────────────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────────────────────────────────────

def test_partita_singola_termina():
    """Una partita deve terminare in round <= 60."""
    risultato = simula_partita(seed=42)
    assert risultato["vincitore"] is not None, "Nessun vincitore!"
    assert risultato["motivo_fine"] is not None
    assert risultato["round_finale"] <= ROUND_CAP_SICUREZZA + 1
    print(f"✓ Partita seed=42: vincitore={risultato['vincitore']}, "
          f"motivo={risultato['motivo_fine']}, "
          f"round={risultato['round_finale']}, "
          f"turni={risultato['turni_giocati']}, "
          f"durata={risultato['durata_secondi']:.2f}s")


def test_molte_partite_terminano():
    """Su 50 partite con seed diversi, tutte devono terminare."""
    risultati = []
    for seed in range(50):
        r = simula_partita(seed=seed)
        risultati.append(r)
        assert r["vincitore"] is not None, f"Seed {seed}: nessun vincitore"

    # Statistiche
    motivi = {}
    for r in risultati:
        motivi[r["motivo_fine"]] = motivi.get(r["motivo_fine"], 0) + 1

    vincitori = {}
    for r in risultati:
        vincitori[r["vincitore"]] = vincitori.get(r["vincitore"], 0) + 1

    durata_media = sum(r["durata_secondi"] for r in risultati) / len(risultati)
    round_medio = sum(r["round_finale"] for r in risultati) / len(risultati)

    print(f"✓ 50 partite tutte terminate")
    print(f"  Motivi fine: {motivi}")
    print(f"  Vincitori: {vincitori}")
    print(f"  Round medio: {round_medio:.1f}")
    print(f"  Durata media: {durata_media:.3f}s")


def test_no_violazioni_invariant():
    """
    Su 20 partite, verifica invarianti che NON devono mai essere violate:
    - Nessun territorio ha proprietario None alla fine
    - Tutti i territori hanno almeno 1 armata
    - Numero totale territori = 42
    - Vincitore è uno dei colori canonici
    """
    for seed in range(20):
        r = simula_partita(seed=seed)
        # Riproduco la partita per controllare lo stato finale
        stato = crea_partita_iniziale(seed=seed)
        rng = random.Random(seed * 7 + 1)
        turni = 0
        while not stato.terminata and turni < 1000:
            colore = stato.giocatore_corrente
            if colore is None or not stato.giocatori[colore].vivo:
                avanza_turno(stato)
                continue
            bot_random_gioca_turno(stato, colore, rng)
            turni += 1
            if stato.terminata:
                break
            gestisci_fine_turno(stato, colore, rng)
            if stato.terminata:
                break
            avanza_turno(stato)

        # Invarianti
        territori_validi = sum(1 for s in stato.mappa.values()
                              if s.proprietario in COLORI_GIOCATORI)
        assert territori_validi == 42, (
            f"Seed {seed}: solo {territori_validi}/42 territori con proprietario valido"
        )
        for t, s in stato.mappa.items():
            assert s.armate >= 1, f"Seed {seed}: {t} ha {s.armate} armate"
        assert stato.vincitore in COLORI_GIOCATORI

    print("✓ 20 partite: invarianti rispettati (42 territori, ≥1 armata, vincitore valido)")


def test_cap_armate_130_mai_superato():
    """Su molte partite, nessun giocatore deve mai superare 130 armate."""
    for seed in range(20):
        stato = crea_partita_iniziale(seed=seed)
        rng = random.Random(seed * 7 + 1)
        turni = 0

        while not stato.terminata and turni < 1000:
            colore = stato.giocatore_corrente
            if colore is None or not stato.giocatori[colore].vivo:
                avanza_turno(stato)
                continue

            bot_random_gioca_turno(stato, colore, rng)
            turni += 1

            # Verifica cap dopo ogni turno
            for col in COLORI_GIOCATORI:
                armate = stato.num_armate_di(col)
                assert armate <= 130, (
                    f"Seed {seed}, round {stato.round_corrente}, "
                    f"{col}: {armate} armate > 130"
                )

            if stato.terminata:
                break
            gestisci_fine_turno(stato, colore, rng)
            if stato.terminata:
                break
            avanza_turno(stato)

    print("✓ 20 partite: cap 130 armate mai superato")


def test_velocita_simulazione():
    """Misura la velocità di simulazione (utile per stimare tempi training)."""
    n_partite = 100
    inizio = time.time()
    for seed in range(n_partite):
        simula_partita(seed=seed)
    durata = time.time() - inizio

    partite_per_secondo = n_partite / durata
    print(f"✓ Velocità: {n_partite} partite in {durata:.1f}s "
          f"= {partite_per_secondo:.0f} partite/sec")
    print(f"  (Stima per 10M partite: {10_000_000 / partite_per_secondo / 3600:.1f} ore)")


def test_riproducibilita_partita():
    """Stesso seed → stessa partita identica (vincitore, motivo, round)."""
    r1 = simula_partita(seed=12345)
    r2 = simula_partita(seed=12345)
    assert r1["vincitore"] == r2["vincitore"]
    assert r1["motivo_fine"] == r2["motivo_fine"]
    assert r1["round_finale"] == r2["round_finale"]
    print(f"✓ Riproducibilità: seed=12345 → "
          f"sempre vincitore={r1['vincitore']}, "
          f"round={r1['round_finale']}")


# ─────────────────────────────────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_partita_singola_termina,
        test_riproducibilita_partita,
        test_molte_partite_terminano,
        test_no_violazioni_invariant,
        test_cap_armate_130_mai_superato,
        test_velocita_simulazione,
    ]

    print("\n" + "=" * 60)
    print("Test Integrazione: Partite complete end-to-end")
    print("=" * 60 + "\n")

    falliti = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} FALLITO: {e}")
            falliti.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} ERRORE: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            falliti.append(test.__name__)

    print("\n" + "=" * 60)
    if falliti:
        print(f"FALLITI: {len(falliti)}/{len(tests)}")
        for nome in falliti:
            print(f"  - {nome}")
    else:
        print(f"TUTTI I {len(tests)} TEST PASSATI ✓")
    print("=" * 60 + "\n")

    return len(falliti) == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
