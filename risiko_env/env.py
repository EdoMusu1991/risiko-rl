"""
env.py — Modulo 5c: Environment Gymnasium per RisiKo.

Trasforma il simulatore in un Gym env standard.
Il bot RL gioca un singolo giocatore (default: BLU).
Gli altri 3 giocatori sono controllati da un bot random integrato.

API Gymnasium:
- reset() → observation, info
- step(action) → observation, reward, terminated, truncated, info

State machine interna: ogni step consuma una decisione del bot in funzione
della fase corrente del suo turno (tris/rinforzo/attacco/continua/quantità/spostamento).

Specifica di riferimento: risiko_specifica_v1.2.md
"""

import random
from typing import Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .data import (
    COLORI_GIOCATORI,
    TUTTI_TERRITORI,
    ADIACENZE,
)
from .stato import StatoPartita, Fase
from .setup import crea_partita_iniziale
from .motore import (
    calcola_rinforzi_base,
    calcola_bonus_continenti,
    seleziona_due_tris_disgiunti,
    calcola_bonus_tris,
    gioca_tris,
    piazza_rinforzi,
    territori_attaccabili_da,
    esegui_attacco,
    applica_conquista,
    spostamento_legale,
    esegui_spostamento,
    pesca_carta,
    avanza_turno,
    EsitoAttacco,
)
from .sdadata import gestisci_fine_turno
from .encoding import (
    codifica_osservazione,
    DIM_OBSERVATION,
    INDEX_TERRITORIO,
)
from .azioni import (
    NUM_AZIONI_TRIS,
    NUM_AZIONI_RINFORZO,
    NUM_AZIONI_ATTACCO,
    NUM_AZIONI_CONTINUA,
    NUM_AZIONI_QUANTITA,
    NUM_AZIONI_SPOSTAMENTO,
    INDICE_STOP_ATTACCO,
    INDICE_SKIP_SPOSTAMENTO,
    enumera_combinazioni_tris,
    maschera_tris,
    maschera_rinforzo,
    maschera_attacco,
    maschera_continua,
    maschera_quantita,
    calcola_quantita_da_azione,
    maschera_spostamento,
    decodifica_azione_attacco,
    decodifica_azione_spostamento,
)


# ═════════════════════════════════════════════════════════════════════════
#  COSTANTI
# ═════════════════════════════════════════════════════════════════════════

# Action space unificato: prende il MAX delle dimensioni per fase
# Cosi possiamo usare uno spazio Discreto singolo. Le maschere ci dicono
# quali indici sono validi in ogni momento.
ACTION_SPACE_SIZE = max(
    NUM_AZIONI_TRIS,
    NUM_AZIONI_RINFORZO,
    NUM_AZIONI_ATTACCO,
    NUM_AZIONI_CONTINUA,
    NUM_AZIONI_QUANTITA,
    NUM_AZIONI_SPOSTAMENTO,
)  # = 1765


# Sotto-fasi della macchina a stati
class SottoFase:
    TRIS = "tris"
    RINFORZO = "rinforzo"           # un'armata alla volta
    ATTACCO = "attacco"             # scegli (da, verso) o stop
    CONTINUA = "continua"           # dopo un lancio, continua o ferma
    QUANTITA_CONQUISTA = "quantita_conquista"  # post-conquista, quante armate spostare
    SPOSTAMENTO = "spostamento"     # spostamento finale (da, verso) o skip
    QUANTITA_SPOSTAMENTO = "quantita_spostamento"  # dopo aver scelto coppia spostamento


# Reward (sparso, a fine partita)
# Indici: posizione_finale → reward
REWARD_PER_POSIZIONE = {
    1: 1.0,    # vittoria
    2: 0.3,    # secondo
    3: -0.3,   # terzo
    4: -1.0,   # quarto/eliminato
}


# ═════════════════════════════════════════════════════════════════════════
#  BOT RANDOM (per gli avversari)
# ═════════════════════════════════════════════════════════════════════════

def _bot_random_gioca_turno(
    stato: StatoPartita,
    colore: str,
    rng: random.Random,
) -> None:
    """
    Esegue un intero turno per `colore` con strategia COMPLETAMENTE RANDOM.

    IMPORTANTE: questo bot deve giocare con la stessa "stupidità" del bot RL
    quando è random (non addestrato). Altrimenti c'è asimmetria: il bot RL
    deve battere avversari smart, ma parte da random — partita persa in partenza.

    Strategia uniforme: per ogni decisione, sceglie azione random tra quelle legali.
    Niente euristiche, niente filtri ratio, niente preferenze di ratio.
    """
    giocatore = stato.giocatori[colore]
    if not stato.territori_di(colore):
        return  # eliminato

    # ── FASE 1: tris e rinforzi ──────────────────────────────────
    # Tris: random se giocarli o no
    tris_da_giocare = seleziona_due_tris_disgiunti(giocatore.carte)
    bonus_tris = 0
    if tris_da_giocare and rng.random() < 0.5:
        bonus_tris = calcola_bonus_tris(stato, colore, tris_da_giocare)
        gioca_tris(stato, colore, tris_da_giocare)

    rinf_base = calcola_rinforzi_base(stato, colore)
    bonus_cont = calcola_bonus_continenti(stato, colore)
    totale_rinforzi = rinf_base + bonus_cont + bonus_tris

    # Cap 130 (come fa il bot RL)
    armate_correnti = stato.num_armate_di(colore)
    spazio = max(0, 130 - armate_correnti)
    totale_rinforzi = min(totale_rinforzi, spazio)

    territori_propri = stato.territori_di(colore)
    if not territori_propri:
        return

    if totale_rinforzi > 0:
        # Distribuzione casuale uniforme sui propri territori
        distribuzione = {}
        for _ in range(totale_rinforzi):
            t = rng.choice(territori_propri)
            distribuzione[t] = distribuzione.get(t, 0) + 1
        piazza_rinforzi(stato, colore, distribuzione)

    # ── FASE 2: attacchi ─────────────────────────────────────────
    # Numero di attacchi random tra 0 e 3 (era 5 con filtri)
    # Niente filtro rapporto: attacca anche se sfavorevole
    max_attacchi = rng.randint(0, 3)
    n_attacchi = 0
    while n_attacchi < max_attacchi and not stato.terminata:
        propri_con_armate = [t for t in stato.territori_di(colore)
                             if stato.mappa[t].armate >= 2]
        if not propri_con_armate:
            break

        da = rng.choice(propri_con_armate)
        attaccabili = territori_attaccabili_da(stato, da)
        if not attaccabili:
            n_attacchi += 1
            continue

        verso = rng.choice(attaccabili)
        # NIENTE filtro rapporto >= 1.5: attacca a ratio random
        # 1 lancio per attacco (come bot RL), poi decide random se continuare
        esito = esegui_attacco(stato, colore, da, verso, rng,
                              fermati_dopo_lanci=1)

        # 50% di chance di continuare a tirare se non ha conquistato
        while (not esito.conquistato
               and not stato.terminata
               and rng.random() < 0.5
               and stato.mappa[da].proprietario == colore
               and stato.mappa[da].armate >= 2
               and stato.mappa[verso].proprietario != colore):
            from .motore import attacco_legale
            if not attacco_legale(stato, colore, da, verso):
                break
            esito = esegui_attacco(stato, colore, da, verso, rng,
                                  fermati_dopo_lanci=1)

        if esito.conquistato:
            minimo = esito.num_dadi_ultimo_lancio
            massimo = stato.mappa[da].armate - 1
            if minimo <= massimo:
                # Quantità random tra minimo e massimo
                quantita = rng.randint(minimo, massimo)
                fine = applica_conquista(stato, colore, da, verso,
                                         quantita, esito, rng)
                if fine:
                    return
        n_attacchi += 1
        if stato.terminata:
            return

    # ── FASE 3: spostamento (random, 30%) ────────────────────────
    if rng.random() < 0.3:
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
            from .motore import _minimo_da_lasciare_per_spostamento
            min_da_lasciare = _minimo_da_lasciare_per_spostamento(stato, da, colore)
            massimo = stato.mappa[da].armate - min_da_lasciare
            if massimo >= 1:
                quantita = rng.randint(1, massimo)
                if spostamento_legale(stato, colore, da, verso, quantita):
                    esegui_spostamento(stato, colore, da, verso, quantita)

    # ── FASE 4: pesca carta ──────────────────────────────────────
    pesca_carta(stato, colore, rng)


# ═════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT GYMNASIUM
# ═════════════════════════════════════════════════════════════════════════

class RisikoEnv(gym.Env):
    """
    Environment Gymnasium per RisiKo.

    Il bot RL controlla un singolo giocatore (default BLU).
    Gli altri 3 sono bot random.

    Observation: vettore float32 di dimensione 318 (vedi encoding.py).
    Action: int in [0, 1765). Solo gli indici nella maschera sono legali.

    Use:
        env = RisikoEnv(seed=42)
        obs, info = env.reset()
        action = ... # scegli da info["action_mask"]
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        bot_color: str = "BLU",
        max_steps: int = 5000,
        seed: Optional[int] = None,
    ):
        """
        Args:
            bot_color: colore controllato dal bot RL
            max_steps: limite step per partita (failsafe contro loop)
            seed: per riproducibilità
        """
        super().__init__()
        assert bot_color in COLORI_GIOCATORI, f"Colore invalido: {bot_color}"
        self.bot_color = bot_color
        self.max_steps = max_steps
        self._initial_seed = seed

        # Spazi Gymnasium
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(DIM_OBSERVATION,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Stato interno (verrà inizializzato in reset)
        self.stato: Optional[StatoPartita] = None
        self.rng: Optional[random.Random] = None
        self.sotto_fase: Optional[str] = None
        self.step_count: int = 0

        # Stato della macchina (variabili usate dalle sotto-fasi)
        self._combinazioni_tris: list = []  # opzioni tris correnti
        self._rinforzi_rimasti: int = 0     # armate ancora da piazzare
        self._attacco_corrente: Optional[tuple[str, str]] = None  # (da, verso) attacco in corso
        self._esito_attacco_corrente: Optional[EsitoAttacco] = None  # esito ultimo lancio
        self._spostamento_corrente: Optional[tuple[str, str]] = None  # (da, verso) spostamento

    # ─────────────────────────────────────────────────────────────
    #  RESET
    # ─────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """
        Reset dell'ambiente. Crea una nuova partita.
        """
        if seed is None:
            seed = self._initial_seed

        # Seed per setup partita
        self.stato = crea_partita_iniziale(seed=seed)
        # Seed diverso per il rng di gioco (evita correlazioni con setup)
        rng_seed = (seed * 7 + 13) if seed is not None else None
        self.rng = random.Random(rng_seed)

        self.step_count = 0
        self.sotto_fase = None
        self._combinazioni_tris = []
        self._rinforzi_rimasti = 0
        self._attacco_corrente = None
        self._esito_attacco_corrente = None
        self._spostamento_corrente = None

        # Avanza fino al primo turno del bot
        self._avanza_fino_a_turno_bot()

        # Calcola observation iniziale
        obs = self._costruisci_observation()
        info = self._costruisci_info()

        return obs, info

    # ─────────────────────────────────────────────────────────────
    #  STEP
    # ─────────────────────────────────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Esegue una decisione del bot e fa avanzare lo stato.
        Restituisce (obs, reward, terminated, truncated, info).
        """
        self.step_count += 1

        # Verifica che siamo nel turno del bot
        assert self.stato.giocatore_corrente == self.bot_color, (
            f"step() chiamato fuori turno bot. "
            f"Corrente: {self.stato.giocatore_corrente}"
        )

        # Snapshot pre-step per reward shaping
        n_terr_pre = self.stato.num_territori_di(self.bot_color)
        n_arm_pre = self.stato.num_armate_di(self.bot_color)

        # Esegui l'azione in base alla sotto-fase
        if self.sotto_fase == SottoFase.TRIS:
            self._step_tris(action)
        elif self.sotto_fase == SottoFase.RINFORZO:
            self._step_rinforzo(action)
        elif self.sotto_fase == SottoFase.ATTACCO:
            self._step_attacco(action)
        elif self.sotto_fase == SottoFase.CONTINUA:
            self._step_continua(action)
        elif self.sotto_fase == SottoFase.QUANTITA_CONQUISTA:
            self._step_quantita_conquista(action)
        elif self.sotto_fase == SottoFase.SPOSTAMENTO:
            self._step_spostamento(action)
        elif self.sotto_fase == SottoFase.QUANTITA_SPOSTAMENTO:
            self._step_quantita_spostamento(action)
        else:
            raise RuntimeError(f"Sotto-fase invalida: {self.sotto_fase}")

        # Se la partita è terminata o il bot ha finito il turno, fa girare gli altri
        if not self.stato.terminata:
            if self.sotto_fase is None:
                # Il bot ha finito il turno → vai al prossimo turno (altri o bot)
                self._fine_turno_bot()
                if not self.stato.terminata:
                    self._avanza_fino_a_turno_bot()

        # Calcola reward
        reward = 0.0
        terminated = self.stato.terminata
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward = self._calcola_reward_finale()
        else:
            # Reward shaping: piccoli incentivi per dare segnale alla rete
            # NB: magnitudo bassa (0.001-0.01) per non sovrastare il reward terminale (±1)
            n_terr_post = self.stato.num_territori_di(self.bot_color)
            n_arm_post = self.stato.num_armate_di(self.bot_color)

            # +0.001 per ogni territorio conquistato in questo step
            delta_terr = n_terr_post - n_terr_pre
            if delta_terr > 0:
                reward += 0.001 * delta_terr
            elif delta_terr < 0:
                # Penalty leggermente più piccola per perdita (non vogliamo bot iper-difensivo)
                reward += 0.0005 * delta_terr  # delta_terr è negativo

        obs = self._costruisci_observation()
        info = self._costruisci_info()

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────
    #  FLUSSO TURNO BOT
    # ─────────────────────────────────────────────────────────────

    def _avanza_fino_a_turno_bot(self) -> None:
        """
        Esegue automaticamente i turni degli altri giocatori finché:
        - tocca al bot, oppure
        - la partita finisce
        """
        while not self.stato.terminata:
            corrente = self.stato.giocatore_corrente
            if corrente is None:
                break

            if not self.stato.giocatori[corrente].vivo:
                avanza_turno(self.stato)
                continue

            if corrente == self.bot_color:
                # Inizia il turno del bot, sotto-fase TRIS
                self._inizia_fase_tris()
                break

            # Turno di un avversario: bot random
            _bot_random_gioca_turno(self.stato, corrente, self.rng)
            if self.stato.terminata:
                break
            gestisci_fine_turno(self.stato, corrente, self.rng)
            if self.stato.terminata:
                break
            avanza_turno(self.stato)

    def _fine_turno_bot(self) -> None:
        """
        Chiamato quando il bot ha finito le sue 4 fasi.
        Esegue pesca carta + sdadata + avanza al prossimo turno.
        """
        # Pesca carta (se conquistato)
        pesca_carta(self.stato, self.bot_color, self.rng)
        # Sdadata e cap di sicurezza
        gestisci_fine_turno(self.stato, self.bot_color, self.rng)
        if not self.stato.terminata:
            avanza_turno(self.stato)

    # ─────────────────────────────────────────────────────────────
    #  SOTTO-FASE TRIS
    # ─────────────────────────────────────────────────────────────

    def _inizia_fase_tris(self) -> None:
        """Inizia la fase tris del bot. Calcola le combinazioni disponibili."""
        carte = self.stato.giocatori[self.bot_color].carte
        self._combinazioni_tris = enumera_combinazioni_tris(carte)
        self.sotto_fase = SottoFase.TRIS

    def _step_tris(self, action: int) -> None:
        """Bot ha scelto quale combinazione di tris giocare (0=skip)."""
        # Valida indice
        if action >= len(self._combinazioni_tris):
            action = 0  # fallback su skip

        scelta = self._combinazioni_tris[action]
        if scelta:
            # Gioca i tris scelti
            gioca_tris(self.stato, self.bot_color, scelta)

        # Calcola rinforzi totali e passa a sotto-fase rinforzo
        bonus_tris = calcola_bonus_tris(self.stato, self.bot_color, scelta) if scelta else 0
        rinf_base = calcola_rinforzi_base(self.stato, self.bot_color)
        bonus_cont = calcola_bonus_continenti(self.stato, self.bot_color)
        self._rinforzi_rimasti = rinf_base + bonus_cont + bonus_tris

        # Cap 130
        armate_correnti = self.stato.num_armate_di(self.bot_color)
        spazio = max(0, 130 - armate_correnti)
        self._rinforzi_rimasti = min(self._rinforzi_rimasti, spazio)

        if self._rinforzi_rimasti > 0:
            self.sotto_fase = SottoFase.RINFORZO
        else:
            # Niente rinforzi, passa direttamente agli attacchi
            self.sotto_fase = SottoFase.ATTACCO

    # ─────────────────────────────────────────────────────────────
    #  SOTTO-FASE RINFORZO
    # ─────────────────────────────────────────────────────────────

    def _step_rinforzo(self, action: int) -> None:
        """Bot ha scelto un territorio dove piazzare 1 armata."""
        # Decodifica indice → nome territorio
        if action < 0 or action >= 42:
            action = 0  # fallback
        territorio = INDEX_TERRITORIO[action]

        # Valida che sia del bot (altrimenti fallback al primo territorio proprio)
        if self.stato.mappa[territorio].proprietario != self.bot_color:
            propri = self.stato.territori_di(self.bot_color)
            if not propri:
                # Bot eliminato? Caso anomalo
                self.sotto_fase = None
                return
            territorio = propri[0]

        # Piazza 1 armata
        piazza_rinforzi(self.stato, self.bot_color, {territorio: 1})
        self._rinforzi_rimasti -= 1

        if self._rinforzi_rimasti <= 0:
            self.sotto_fase = SottoFase.ATTACCO

    # ─────────────────────────────────────────────────────────────
    #  SOTTO-FASE ATTACCO
    # ─────────────────────────────────────────────────────────────

    def _step_attacco(self, action: int) -> None:
        """Bot sceglie un attacco (da, verso) o stop."""
        if action == INDICE_STOP_ATTACCO or action >= NUM_AZIONI_ATTACCO:
            # Bot ha scelto di fermarsi
            self.sotto_fase = SottoFase.SPOSTAMENTO
            return

        decoded = decodifica_azione_attacco(action)
        if decoded is None:
            self.sotto_fase = SottoFase.SPOSTAMENTO
            return

        da, verso = decoded
        # Valida l'attacco (se non è legale, salta a fase spostamento)
        from .motore import attacco_legale
        if not attacco_legale(self.stato, self.bot_color, da, verso):
            self.sotto_fase = SottoFase.SPOSTAMENTO
            return

        # Esegui un singolo lancio di dadi
        self._attacco_corrente = (da, verso)
        self._esito_attacco_corrente = esegui_attacco(
            self.stato, self.bot_color, da, verso, self.rng,
            fermati_dopo_lanci=1,
        )

        if self._esito_attacco_corrente.conquistato:
            # Conquista! Il bot deve scegliere quante armate spostare
            self.sotto_fase = SottoFase.QUANTITA_CONQUISTA
        else:
            # Il bot decide se continuare o fermarsi
            self.sotto_fase = SottoFase.CONTINUA

    # ─────────────────────────────────────────────────────────────
    #  SOTTO-FASE CONTINUA
    # ─────────────────────────────────────────────────────────────

    def _step_continua(self, action: int) -> None:
        """0=stop, 1=continua a tirare i dadi sullo stesso attacco."""
        if action == 0 or self._attacco_corrente is None:
            # Stop: torna a sotto-fase ATTACCO per scegliere altro
            self._attacco_corrente = None
            self._esito_attacco_corrente = None
            self.sotto_fase = SottoFase.ATTACCO
            return

        # Continua: altro lancio
        da, verso = self._attacco_corrente
        # Verifica legalità (potrebbe essere cambiato)
        from .motore import attacco_legale
        if not attacco_legale(self.stato, self.bot_color, da, verso):
            self._attacco_corrente = None
            self._esito_attacco_corrente = None
            self.sotto_fase = SottoFase.ATTACCO
            return

        # Esegui un altro lancio
        nuovo_esito = esegui_attacco(
            self.stato, self.bot_color, da, verso, self.rng,
            fermati_dopo_lanci=1,
        )
        # Aggiorna l'esito complessivo (le armate finali sono quelle di nuovo_esito)
        # Se conquistato → quantità
        if nuovo_esito.conquistato:
            self._esito_attacco_corrente = nuovo_esito
            self.sotto_fase = SottoFase.QUANTITA_CONQUISTA
        else:
            # Decide ancora continua/stop
            self._esito_attacco_corrente = nuovo_esito
            self.sotto_fase = SottoFase.CONTINUA

    # ─────────────────────────────────────────────────────────────
    #  SOTTO-FASE QUANTITA CONQUISTA
    # ─────────────────────────────────────────────────────────────

    def _step_quantita_conquista(self, action: int) -> None:
        """Bot sceglie quante armate spostare (0=min, 1=mid, 2=max)."""
        assert self._attacco_corrente is not None
        assert self._esito_attacco_corrente is not None

        da, verso = self._attacco_corrente
        esito = self._esito_attacco_corrente

        minimo = esito.num_dadi_ultimo_lancio
        massimo = self.stato.mappa[da].armate - 1

        if minimo > massimo:
            # Caso degenerato: non c'è modo di spostare valido. Forza stop.
            self._attacco_corrente = None
            self._esito_attacco_corrente = None
            self.sotto_fase = SottoFase.ATTACCO
            return

        action = max(0, min(2, action))
        quantita = calcola_quantita_da_azione(action, minimo, massimo)

        # Applica la conquista
        fine = applica_conquista(
            self.stato, self.bot_color, da, verso, quantita, esito, self.rng,
        )
        # Reset variabili attacco
        self._attacco_corrente = None
        self._esito_attacco_corrente = None

        if fine:
            # Vittoria immediata per obiettivo
            self.sotto_fase = None
            return

        # Torna a sotto-fase ATTACCO per scegliere il prossimo attacco
        self.sotto_fase = SottoFase.ATTACCO

    # ─────────────────────────────────────────────────────────────
    #  SOTTO-FASE SPOSTAMENTO
    # ─────────────────────────────────────────────────────────────

    def _step_spostamento(self, action: int) -> None:
        """Bot sceglie (da, verso) per spostamento finale o skip."""
        if action == INDICE_SKIP_SPOSTAMENTO or action >= NUM_AZIONI_SPOSTAMENTO:
            # Skip: turno finito
            self.sotto_fase = None
            return

        decoded = decodifica_azione_spostamento(action)
        if decoded is None:
            self.sotto_fase = None
            return

        da, verso = decoded
        # Valida
        if (self.stato.mappa[da].proprietario != self.bot_color
                or self.stato.mappa[verso].proprietario != self.bot_color
                or verso not in ADIACENZE[da]):
            self.sotto_fase = None
            return

        # Memorizza per la quantità
        self._spostamento_corrente = (da, verso)
        self.sotto_fase = SottoFase.QUANTITA_SPOSTAMENTO

    def _step_quantita_spostamento(self, action: int) -> None:
        """Bot sceglie quante armate spostare nello spostamento finale."""
        assert self._spostamento_corrente is not None
        da, verso = self._spostamento_corrente

        # Calcola minimo e massimo
        from .motore import _minimo_da_lasciare_per_spostamento
        min_lasciare = _minimo_da_lasciare_per_spostamento(self.stato, da, self.bot_color)
        massimo = self.stato.mappa[da].armate - min_lasciare

        if massimo < 1:
            # Caso degenerato
            self._spostamento_corrente = None
            self.sotto_fase = None
            return

        minimo = 1  # in spostamento finale, minimo è 1 armata da spostare
        action = max(0, min(2, action))
        quantita = calcola_quantita_da_azione(action, minimo, massimo)

        if spostamento_legale(self.stato, self.bot_color, da, verso, quantita):
            esegui_spostamento(self.stato, self.bot_color, da, verso, quantita)

        self._spostamento_corrente = None
        self.sotto_fase = None  # turno finito

    # ─────────────────────────────────────────────────────────────
    #  OBSERVATION & ACTION MASK
    # ─────────────────────────────────────────────────────────────

    def _costruisci_observation(self) -> np.ndarray:
        """Costruisce l'observation dal POV del bot."""
        if self.stato is None:
            return np.zeros(DIM_OBSERVATION, dtype=np.float32)

        fase_corrente = self._fase_corrente_per_encoding()
        return codifica_osservazione(self.stato, self.bot_color, fase_corrente)

    def _fase_corrente_per_encoding(self) -> str:
        """Mappa sotto-fase → fase principale (per encoding)."""
        mapping = {
            SottoFase.TRIS: Fase.TRIS_E_RINFORZI,
            SottoFase.RINFORZO: Fase.TRIS_E_RINFORZI,
            SottoFase.ATTACCO: Fase.ATTACCHI,
            SottoFase.CONTINUA: Fase.ATTACCHI,
            SottoFase.QUANTITA_CONQUISTA: Fase.ATTACCHI,
            SottoFase.SPOSTAMENTO: Fase.SPOSTAMENTO,
            SottoFase.QUANTITA_SPOSTAMENTO: Fase.SPOSTAMENTO,
        }
        return mapping.get(self.sotto_fase, Fase.TRIS_E_RINFORZI)

    def get_action_mask(self) -> np.ndarray:
        """
        Restituisce la maschera delle azioni legali nello stato corrente.
        Per MaskablePPO.
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

        if self.stato is None or self.stato.terminata:
            mask[0] = True  # azione no-op se finito
            return mask

        if self.sotto_fase == SottoFase.TRIS:
            sub = maschera_tris(self._combinazioni_tris)
            mask[:len(sub)] = sub
        elif self.sotto_fase == SottoFase.RINFORZO:
            sub = maschera_rinforzo(self.stato, self.bot_color)
            mask[:len(sub)] = sub
        elif self.sotto_fase == SottoFase.ATTACCO:
            sub = maschera_attacco(self.stato, self.bot_color)
            mask[:len(sub)] = sub
        elif self.sotto_fase == SottoFase.CONTINUA:
            if self._attacco_corrente is not None:
                da, verso = self._attacco_corrente
                sub = maschera_continua(self.stato, self.bot_color, da, verso)
                mask[:len(sub)] = sub
            else:
                mask[0] = True  # solo stop
        elif self.sotto_fase == SottoFase.QUANTITA_CONQUISTA:
            if self._attacco_corrente is not None and self._esito_attacco_corrente is not None:
                da, _ = self._attacco_corrente
                minimo = self._esito_attacco_corrente.num_dadi_ultimo_lancio
                massimo = self.stato.mappa[da].armate - 1
                sub = maschera_quantita(minimo, max(minimo, massimo))
                mask[:len(sub)] = sub
            else:
                mask[0] = True
        elif self.sotto_fase == SottoFase.SPOSTAMENTO:
            sub = maschera_spostamento(self.stato, self.bot_color)
            mask[:len(sub)] = sub
        elif self.sotto_fase == SottoFase.QUANTITA_SPOSTAMENTO:
            if self._spostamento_corrente is not None:
                da, _ = self._spostamento_corrente
                from .motore import _minimo_da_lasciare_per_spostamento
                min_lasc = _minimo_da_lasciare_per_spostamento(
                    self.stato, da, self.bot_color
                )
                massimo = self.stato.mappa[da].armate - min_lasc
                if massimo >= 1:
                    sub = maschera_quantita(1, massimo)
                    mask[:len(sub)] = sub
                else:
                    mask[0] = True
            else:
                mask[0] = True
        else:
            mask[0] = True

        # Failsafe: almeno un'azione deve essere legale
        if not mask.any():
            mask[0] = True

        return mask

    def _costruisci_info(self) -> dict:
        """Dizionario info con action_mask, sotto_fase, ecc."""
        info = {
            "action_mask": self.get_action_mask(),
            "sotto_fase": self.sotto_fase,
            "round": self.stato.round_corrente if self.stato else 0,
            "vincitore": self.stato.vincitore if self.stato else None,
            "motivo_fine": self.stato.motivo_fine if self.stato else None,
        }
        return info

    # ─────────────────────────────────────────────────────────────
    #  REWARD
    # ─────────────────────────────────────────────────────────────

    def _calcola_reward_finale(self) -> float:
        """
        Reward sparso a fine partita basato sulla posizione del bot.
        +1 se vince, -1 se eliminato, valori intermedi per posizioni intermedie.
        """
        if self.stato.vincitore == self.bot_color:
            return REWARD_PER_POSIZIONE[1]

        # Bot eliminato
        if not self.stato.giocatori[self.bot_color].vivo:
            return REWARD_PER_POSIZIONE[4]

        # Calcola posizione finale tra giocatori vivi (per punti in obiettivo)
        from .obiettivi import calcola_punti_in_obiettivo
        punti = {
            col: calcola_punti_in_obiettivo(self.stato, col)
            for col in COLORI_GIOCATORI
            if self.stato.giocatori[col].vivo
        }
        # Ordina decrescente per punti
        ordinati = sorted(punti.items(), key=lambda x: x[1], reverse=True)
        for posizione, (col, _) in enumerate(ordinati, start=1):
            if col == self.bot_color:
                return REWARD_PER_POSIZIONE.get(posizione, 0.0)
        return 0.0

    # ─────────────────────────────────────────────────────────────
    #  RENDER
    # ─────────────────────────────────────────────────────────────

    def render(self, mode: str = "human") -> Optional[str]:
        """Render minimale: stampa lo stato come testo."""
        if self.stato is None:
            return None
        lines = []
        lines.append(f"Round: {self.stato.round_corrente} | "
                     f"Turno: {self.stato.giocatore_corrente} | "
                     f"Sotto-fase: {self.sotto_fase}")
        for col in COLORI_GIOCATORI:
            g = self.stato.giocatori[col]
            tag = "✗" if not g.vivo else "  "
            lines.append(f"  {tag} {col}: {self.stato.num_territori_di(col)} terr, "
                         f"{self.stato.num_armate_di(col)} arm, "
                         f"{g.num_carte()} carte")
        if self.stato.terminata:
            lines.append(f"FINE: vincitore={self.stato.vincitore}, "
                         f"motivo={self.stato.motivo_fine}")
        text = "\n".join(lines)

        if mode == "human":
            print(text)
            return None
        return text
