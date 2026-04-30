"""
visualizza_partita.py — Visualizza una partita giocata dal bot con commento turno-per-turno.

Carica un modello, gioca una partita contro 3 random, e stampa il replay
con tutti gli eventi: attacchi del bot, attacchi avversari, conquiste,
stato della partita ad intervalli regolari.

Uso:
    python scripts/visualizza_partita.py modello.zip
    python scripts/visualizza_partita.py modello.zip --seed 42
    python scripts/visualizza_partita.py modello.zip --bot_color ROSSO
    python scripts/visualizza_partita.py modello.zip --solo_bot   # solo eventi del bot

Senza modello (giocatore RANDOM puro, per testare il visualizzatore):
    python scripts/visualizza_partita.py --random --seed 42
"""

import argparse
import sys
import os
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risiko_env import RisikoEnv
from risiko_env.data import COLORI_GIOCATORI, CONTINENTI


# ─────────────────────────────────────────────────────────────────────────
#  COLORI ANSI (per terminali che li supportano)
# ─────────────────────────────────────────────────────────────────────────

class C:
    """Codici colore ANSI."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLU = "\033[94m"
    ROSSO = "\033[91m"
    VERDE = "\033[92m"
    GIALLO = "\033[93m"
    GRIGIO = "\033[37m"


def colora(testo: str, colore: str) -> str:
    """Colora testo con ANSI (se terminale supporta)."""
    map_colore = {
        "BLU": C.BLU, "ROSSO": C.ROSSO, "VERDE": C.VERDE, "GIALLO": C.GIALLO,
    }
    if colore in map_colore:
        return f"{map_colore[colore]}{testo}{C.RESET}"
    return testo


# ─────────────────────────────────────────────────────────────────────────
#  STAMPA EVENTI
# ─────────────────────────────────────────────────────────────────────────

def stampa_tris_e_calcolo(e: dict, bot_color: str) -> None:
    """Stampa tris giocati e calcolo rinforzi del bot."""
    bot_str = colora(bot_color, bot_color)
    rinf_base = e["rinforzi_base"]
    bonus_cont = e["bonus_continenti"]
    bonus_tris = e["bonus_tris"]
    totale = e["totale_rinforzi"]
    if e["tris_giocato"]:
        print(f"  {bot_str} 🃏 gioca {e['num_tris']} tris (+{bonus_tris} carri) | "
              f"base:{rinf_base} continenti:+{bonus_cont} = {totale} totali")
    else:
        if rinf_base + bonus_cont != totale:
            return  # niente di interessante
        if bonus_cont > 0:
            print(f"  {bot_str} 📊 rinforzi: {rinf_base} base + {bonus_cont} continenti = {totale}")
        # Se non ha bonus continenti, info poco interessante: silenzio


def stampa_spostamento_bot(e: dict, bot_color: str) -> None:
    """Stampa lo spostamento finale del bot."""
    bot_str = colora(bot_color, bot_color)
    print(f"  {bot_str} 🔄 sposta {e['quantita']} carri: {e['da']} → {e['verso']}")


def stampa_carta_pescata(e: dict, bot_color: str) -> None:
    """Stampa pesca carta del bot."""
    bot_str = colora(bot_color, bot_color)
    print(f"  {bot_str} 🎴 pesca carta (totale carte: {e['n_carte']})")


def stampa_sdadata(e: dict) -> None:
    """Stampa attivazione sdadata."""
    if e["terminata"]:
        print(f"  {C.BOLD}🔥 PARTITA FINITA{C.RESET} (motivo: {e['motivo_fine']})")
    else:
        print(f"  {C.DIM}⚡ sdadata/cap attivata al round {e['round']}{C.RESET}")


def stampa_rinforzo_bot(e: dict, bot_color: str) -> None:
    """Stampa la distribuzione dei rinforzi del bot."""
    distrib = e["distribuzione"]
    totale = e["totale"]
    bot_str = colora(bot_color, bot_color)
    # Top 3 territori per quantità
    top = sorted(distrib.items(), key=lambda x: -x[1])
    parti = [f"{t}({n})" for t, n in top[:5]]
    if len(top) > 5:
        parti.append(f"+altri {len(top)-5}")
    print(f"  {bot_str} 🔧 piazza {totale} carri: {', '.join(parti)}")


def stampa_attacco_bot(e: dict, bot_color: str) -> None:
    """Stampa un attacco del bot."""
    da = e["da"]
    verso = e["verso"]
    a_pre = e["armate_da_pre"]
    v_pre = e["armate_verso_pre"]
    a_post = e["armate_da_post"]
    v_post = e["armate_verso_post"]
    vittima = e["vittima"]
    conquistato = e["conquistato"]

    perse_attacc = a_pre - a_post
    perse_difens = v_pre - v_post

    bot_str = colora(bot_color, bot_color)
    vit_str = colora(vittima, vittima)
    verbo = "⚔️  attacca"

    if conquistato:
        risultato = f"  → {C.BOLD}CONQUISTA!{C.RESET}"
    else:
        risultato = f"  → respinto ({a_post}vs{v_post})"

    print(f"  {bot_str} {verbo} {vit_str}: {da} ({a_pre}) → {verso} ({v_pre}) | "
          f"perdite: -{perse_attacc} vs -{perse_difens}{risultato}")


def stampa_turno_avversario(e: dict, bot_color: str) -> None:
    """Stampa il riassunto di un turno avversario."""
    avv = e["colore"]
    avv_str = colora(avv, avv)
    if not e["attaccato"]:
        print(f"  {avv_str}: turno passivo (nessun attacco)")
        return

    n_terr_persi = len(e["territori_persi"])
    n_terr_guad = len(e["territori_guadagnati"])
    contro_pov = e["attacchi_contro_pov"]

    msg = f"  {avv_str}: "
    parti = []
    if n_terr_guad > 0:
        terr_str = ", ".join(e["territori_guadagnati"][:3])
        if len(e["territori_guadagnati"]) > 3:
            terr_str += f", +{len(e['territori_guadagnati'])-3}..."
        parti.append(f"conquista {n_terr_guad} ({terr_str})")
    if n_terr_persi > 0:
        parti.append(f"perde {n_terr_persi} territori")
    if contro_pov > 0:
        bot_str = colora(bot_color, bot_color)
        parti.append(f"⚠️  {C.BOLD}attacca te ({contro_pov}x){C.RESET}")
    if not parti:
        parti.append("attacchi senza conquiste")
    print(msg + ", ".join(parti))


def stampa_riassunto_round(stato, round_num: int, bot_color: str) -> None:
    """Stampa lo stato di tutti i giocatori a fine round."""
    print(f"\n  {C.DIM}── Stato dopo round {round_num} ──{C.RESET}")
    for c in COLORI_GIOCATORI:
        g = stato.giocatori[c]
        if not g.vivo:
            print(f"  {colora(c, c)}: {C.DIM}eliminato{C.RESET}")
            continue
        n_terr = stato.num_territori_di(c)
        n_arm = stato.num_armate_di(c)
        n_carte = g.num_carte()
        n_continenti = sum(
            1 for cont, terrs in CONTINENTI.items()
            if all(stato.mappa[t].proprietario == c for t in terrs)
        )
        marker = "👈 BOT" if c == bot_color else ""
        print(f"  {colora(c, c)}: {n_terr:2d} terr, {n_arm:3d} arm, "
              f"{n_carte} carte, {n_continenti} continenti {marker}")


# ─────────────────────────────────────────────────────────────────────────
#  PARTITA
# ─────────────────────────────────────────────────────────────────────────

def gioca_e_visualizza(
    model,
    seed: int = 42,
    bot_color: str = "BLU",
    solo_bot: bool = False,
    intervallo_riassunto: int = 5,
) -> None:
    """Gioca una partita e stampa replay completo."""

    print(f"\n{C.BOLD}{'═' * 70}{C.RESET}")
    print(f"{C.BOLD}  PARTITA: bot={colora(bot_color, bot_color)}, seed={seed}{C.RESET}")
    print(f"{C.BOLD}{'═' * 70}{C.RESET}\n")

    env = RisikoEnv(seed=seed, bot_color=bot_color, log_eventi=True)
    obs, info = env.reset()
    n_step = 0

    while True:
        mask = info["action_mask"]
        if model is None:
            # Bot random per test
            legali = np.where(mask)[0]
            action = np.random.choice(legali)
        else:
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        n_step += 1
        if term or trunc:
            break

    # Adesso processa eventi raggruppandoli per round
    eventi_per_round = defaultdict(list)
    for e in env._eventi:
        eventi_per_round[e["round"]].append(e)

    last_round_stampato = 0
    last_round_eventi = sorted(eventi_per_round.keys())

    for r in last_round_eventi:
        eventi_r = eventi_per_round[r]

        # Header round
        print(f"\n{C.BOLD}── Round {r} ──{C.RESET}")

        # Raggruppa per giocatore di turno
        per_giocatore = defaultdict(list)
        for e in eventi_r:
            per_giocatore[e["turno_di"]].append(e)

        for giocatore_turno, eventi_g in per_giocatore.items():
            if solo_bot and giocatore_turno != bot_color:
                continue

            if giocatore_turno == bot_color:
                # Tutti gli eventi del turno bot, nell'ordine cronologico
                tris_eventi = [e for e in eventi_g if e["tipo"] == "tris_e_calcolo_rinforzi"]
                rinforzi = [e for e in eventi_g if e["tipo"] == "rinforzo_bot"]
                attacchi = [e for e in eventi_g if e["tipo"] == "attacco_bot"]
                spostamenti = [e for e in eventi_g if e["tipo"] == "spostamento_bot"]
                carte = [e for e in eventi_g if e["tipo"] == "carta_pescata"]
                sdadate = [e for e in eventi_g if e["tipo"] == "sdadata_o_cap"]

                if any([tris_eventi, rinforzi, attacchi, spostamenti, carte, sdadate]):
                    n_conq = sum(1 for e in attacchi if e["conquistato"])
                    print(f"\n  Turno bot ({colora(bot_color, bot_color)}): "
                          f"{len(attacchi)} attacchi, {n_conq} conquiste")

                # Ordine cronologico: tris → rinforzi → attacchi → spostamento → carta → sdadata
                for e in tris_eventi:
                    stampa_tris_e_calcolo(e, bot_color)
                for e in rinforzi:
                    stampa_rinforzo_bot(e, bot_color)
                for e in attacchi:
                    stampa_attacco_bot(e, bot_color)
                for e in spostamenti:
                    stampa_spostamento_bot(e, bot_color)
                for e in carte:
                    stampa_carta_pescata(e, bot_color)
                for e in sdadate:
                    stampa_sdadata(e)
            else:
                # Riassunto turno avversario
                turni = [e for e in eventi_g if e["tipo"] == "turno_avversario"]
                for e in turni:
                    stampa_turno_avversario(e, bot_color)

        # Riassunto periodico
        if r % intervallo_riassunto == 0 and r != last_round_stampato:
            stampa_riassunto_round(env.stato, r, bot_color)
            last_round_stampato = r

    # Risultato finale
    print(f"\n{C.BOLD}{'═' * 70}{C.RESET}")
    vincitore = info.get("vincitore")
    motivo = info.get("motivo_fine")
    if vincitore == bot_color:
        print(f"{C.BOLD}  🏆 VITTORIA {colora(bot_color, bot_color)}!{C.RESET} "
              f"(motivo: {motivo})")
    else:
        print(f"{C.BOLD}  Vincitore: {colora(vincitore, vincitore)}{C.RESET} "
              f"(motivo: {motivo})")
    print(f"  Round finali: {info.get('round')}")
    print(f"  Step totali del bot: {n_step}")
    print(f"  Reward bot: {reward:+.2f}")

    # Stats finali
    stato = env.stato
    print(f"\n  {C.BOLD}Stato finale:{C.RESET}")
    for c in COLORI_GIOCATORI:
        g = stato.giocatori[c]
        if not g.vivo:
            print(f"    {colora(c, c)}: {C.DIM}eliminato{C.RESET}")
            continue
        from risiko_env.obiettivi import calcola_punti_in_obiettivo
        punti = calcola_punti_in_obiettivo(stato, c)
        marker = "👈 BOT" if c == bot_color else ""
        print(f"    {colora(c, c)}: {stato.num_territori_di(c):2d} terr, "
              f"{stato.num_armate_di(c):3d} arm, {punti} punti obj {marker}")

    # Riassunto eventi
    print(f"\n  {C.DIM}Riassunto eventi:{C.RESET}")
    tipi = Counter(e["tipo"] for e in env._eventi)
    for t, n in tipi.most_common():
        print(f"    {t}: {n}")
    print()


# ─────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualizza una partita commentata del bot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("modello", nargs="?", default=None,
                        help="Path al .zip del modello (opzionale con --random)")
    parser.add_argument("--random", action="store_true",
                        help="Usa bot random invece di modello (per test)")
    parser.add_argument("--seed", type=int, default=42, help="Seed partita")
    parser.add_argument("--bot_color", default="BLU",
                        choices=COLORI_GIOCATORI,
                        help="Colore del bot (default BLU)")
    parser.add_argument("--solo_bot", action="store_true",
                        help="Mostra solo eventi del bot (silenzia avversari)")
    parser.add_argument("--intervallo", type=int, default=5,
                        help="Riassunto stato ogni N round (default 5)")
    args = parser.parse_args()

    if args.random:
        model = None
        print("Modalità RANDOM: bot pesca azioni a caso.")
    else:
        if not args.modello:
            print("ERRORE: serve un modello (.zip) o --random")
            sys.exit(1)
        if not os.path.exists(args.modello):
            print(f"ERRORE: file non trovato: {args.modello}")
            sys.exit(1)
        from _helpers import carica_modello_con_autodetect
        print(f"Carico modello: {args.modello}")
        model = carica_modello_con_autodetect(args.modello)

    gioca_e_visualizza(
        model,
        seed=args.seed,
        bot_color=args.bot_color,
        solo_bot=args.solo_bot,
        intervallo_riassunto=args.intervallo,
    )


if __name__ == "__main__":
    main()
