"""
_helpers.py — Funzioni di supporto condivise tra gli script.

Auto-detect compatibilità Stage A: rileva se un modello salvato è stato
trainato con observation a 318 feature (pre-Stage A) o 330 (Stage A attivo)
e configura l'env di conseguenza.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def carica_modello_con_autodetect(path: str, verbose: bool = True):
    """
    Carica un modello MaskablePPO e auto-configura l'env per matchare
    la sua dimensione observation.

    Returns:
        modello caricato

    Side effect: imposta `risiko_env.encoding.STAGE_A_ATTIVO` in base al
    modello, così le observation generate hanno la dimensione giusta.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        raise ImportError("Installa: pip install sb3-contrib torch")

    model = MaskablePPO.load(path)
    dim_modello = model.observation_space.shape[0]

    from risiko_env import encoding as _encoding

    if dim_modello == 318:
        _encoding.STAGE_A_ATTIVO = False
        if verbose:
            print(f"⚠ Modello con observation a 318 feature (PRE-Stage A). "
                  f"Stage A disabilitato per compatibilità.")
    elif dim_modello == 330:
        _encoding.STAGE_A_ATTIVO = True
        if verbose:
            print(f"✓ Modello con observation a 330 feature (Stage A attivo).")
    else:
        if verbose:
            print(f"⚠ Modello con dimensione observation inattesa: {dim_modello}")
            print(f"  L'env userà la sua dimensione di default.")

    return model
