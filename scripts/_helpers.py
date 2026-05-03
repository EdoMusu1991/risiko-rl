"""
_helpers.py — Funzioni di supporto condivise tra gli script.

Auto-detect compatibilità Stage A: rileva se un modello salvato è stato
trainato con observation a 318 feature (pre-Stage A) o 330 (Stage A attivo)
e configura l'env di conseguenza.
"""

import sys
import os
import io


def force_utf8_output():
    """
    Forza UTF-8 sullo stdout/stderr su Windows. Necessario per stampare
    emoji e barre Unicode (█, ░, ▶, ⚔️ ...) senza errori.

    Da chiamare all'inizio di ogni script che stampa caratteri non-ASCII.
    """
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                          errors='replace', line_buffering=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                          errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass


# Auto-applica il fix all'import
force_utf8_output()


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def carica_modello_con_autodetect(path: str, verbose: bool = True):
    """
    Carica un modello MaskablePPO e auto-configura l'env per matchare
    la sua dimensione observation.

    Se path è una directory, cerca prioritariamente best_model.zip dentro
    (convenzione per i nuovi training con MaskableEvalCallback).

    Returns:
        modello caricato

    Side effect: imposta `risiko_env.encoding.STAGE_A_ATTIVO` in base al
    modello, così le observation generate hanno la dimensione giusta.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        raise ImportError("Installa: pip install sb3-contrib torch")

    # Se è una cartella, cerca best_model.zip dentro
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "best_model.zip"),
            os.path.join(path, "best", "best_model.zip"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                if verbose:
                    print(f"[INFO] Trovato best_model in {cand}")
                path = cand
                break

    model = MaskablePPO.load(path)
    dim_modello = model.observation_space.shape[0]

    from risiko_env import encoding as _encoding

    if dim_modello == 318:
        _encoding.STAGE_A_ATTIVO = False
        if verbose:
            print(f"[!] Modello con observation a 318 feature (PRE-Stage A). "
                  f"Stage A disabilitato per compatibilita.")
    elif dim_modello == 330:
        # Stage A v1 (versione fallita, 4 feature × 3 avversari).
        # NON SUPPORTATO da Stage A2: feature diverse, modello non utilizzabile.
        if verbose:
            print(f"[!] Modello con observation a 330 feature (Stage A v1 deprecato). "
                  f"Compatibilita non garantita con codice Stage A2.")
        _encoding.STAGE_A_ATTIVO = True
    elif dim_modello == 342:
        # Stage A2 (8 feature di stato × 3 avversari)
        _encoding.STAGE_A_ATTIVO = True
        if verbose:
            print(f"[OK] Modello con observation a 342 feature (Stage A2 attivo).")
    else:
        if verbose:
            print(f"[!] Modello con dimensione observation inattesa: {dim_modello}")
            print(f"    L'env usera la sua dimensione di default.")

    return model
