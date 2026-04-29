"""
combattimento.py — Logica dei dadi nel combattimento RisiKo.

Contiene SOLO le primitive del lancio dadi e del loro confronto.
Niente stato di partita, niente territori. Funzioni pure, facili da testare.

La logica di "chi può attaccare chi", "cosa succede dopo la conquista", ecc.
è in motore.py, che usa queste primitive.

Specifica di riferimento: risiko_specifica_v1.2.md sezioni 4.2.3, 4.2.4.
"""

import random


# ─────────────────────────────────────────────────────────────────────────
#  CALCOLO NUMERO DADI
# ─────────────────────────────────────────────────────────────────────────

def num_dadi_attaccante(armate_attaccante: int) -> int:
    """
    Numero di dadi che l'attaccante DEVE tirare.
    Forzato: non è una scelta del giocatore.

    Specifica 4.2.3:
    - 4+ armate → 3 dadi
    - 3 armate → 2 dadi
    - 2 armate → 1 dado
    - 1 armata → 0 (impossibile attaccare)
    """
    if armate_attaccante < 2:
        return 0
    return min(3, armate_attaccante - 1)


def num_dadi_difensore(armate_difensore: int) -> int:
    """
    Numero di dadi che il difensore DEVE tirare.
    Forzato: sempre il massimo possibile.

    Specifica 4.2.3:
    - 3+ armate → 3 dadi
    - 2 armate → 2 dadi
    - 1 armata → 1 dado
    """
    if armate_difensore < 1:
        return 0
    return min(3, armate_difensore)


# ─────────────────────────────────────────────────────────────────────────
#  LANCIO DADI
# ─────────────────────────────────────────────────────────────────────────

def lancia_dadi(numero: int, rng: random.Random) -> list[int]:
    """
    Tira `numero` dadi a 6 facce e restituisce i risultati ORDINATI in modo decrescente.

    Esempio: numero=3 → [5, 3, 2] (sempre dal più alto al più basso).
    """
    dadi = [rng.randint(1, 6) for _ in range(numero)]
    dadi.sort(reverse=True)
    return dadi


# ─────────────────────────────────────────────────────────────────────────
#  RISOLUZIONE DI UN LANCIO
# ─────────────────────────────────────────────────────────────────────────

def risolvi_lancio(
    armate_attaccante: int,
    armate_difensore: int,
    rng: random.Random,
) -> tuple[int, int, list[int], list[int]]:
    """
    Risolve UN singolo lancio di dadi tra attaccante e difensore.

    Restituisce una tupla:
        (perdite_attaccante, perdite_difensore, dadi_att, dadi_dif)

    dove dadi_att e dadi_dif sono ordinati decrescenti (utili per logging/debug).

    Specifica 4.2.4:
    - Si confrontano min(num_att, num_dif) coppie ordinate decrescenti
    - Se attaccante > difensore → difensore perde 1 armata
    - Se attaccante <= difensore (parità inclusa) → attaccante perde 1 armata
    """
    n_att = num_dadi_attaccante(armate_attaccante)
    n_dif = num_dadi_difensore(armate_difensore)

    if n_att == 0 or n_dif == 0:
        # Combattimento impossibile (uno dei due non può tirare)
        return (0, 0, [], [])

    dadi_att = lancia_dadi(n_att, rng)
    dadi_dif = lancia_dadi(n_dif, rng)

    perdite_att = 0
    perdite_dif = 0
    confronti = min(n_att, n_dif)

    for i in range(confronti):
        if dadi_att[i] > dadi_dif[i]:
            perdite_dif += 1
        else:
            # Parità o difensore vince → attaccante perde
            perdite_att += 1

    return (perdite_att, perdite_dif, dadi_att, dadi_dif)
