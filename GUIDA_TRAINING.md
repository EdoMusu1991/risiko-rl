# Guida al training del bot su Colab Pro

Questa guida ti porta passo-passo dall'avere il simulatore al primo bot trainato. Il workflow è basato su **Git + GitHub**: pubblichi il progetto su GitHub, Colab lo clona ad ogni sessione. Più pulito che caricare manualmente i file su Drive.

## Prerequisiti

- Account **Colab Pro** attivo (~$10/mese, GPU T4 disponibile)
- Account **Google Drive** con almeno 5 GB liberi (per i checkpoint)
- Account **GitHub** (gratuito)
- Tutti i **136 test** del simulatore passano sul tuo PC

## Step 1: Crea il repo su GitHub

1. Vai su [github.com/new](https://github.com/new)
2. Crea un nuovo repository:
   - **Nome:** `risiko-rl` (o quello che vuoi)
   - **Privato** va benissimo
   - **NON** spuntare "Initialize with README"
3. Copia l'URL HTTPS del repo (es. `https://github.com/EdoMusu1991/risiko-rl.git`)

## Step 2: Push del progetto su GitHub

Da PowerShell, dentro la cartella `risiko-rl/`:

```powershell
# Inizializza git (se non l'hai gia fatto)
git init
git branch -M main

# .gitignore minimo
@"
.venv/
__pycache__/
*.pyc
*.zip
checkpoints/
tensorboard_logs/
"@ | Out-File -Encoding utf8 .gitignore

# Primo commit
git add .
git commit -m "Simulatore RisiKo completo (Moduli 1-5c, 136 test)"

# Collega al tuo repo (sostituisci con il tuo URL)
git remote add origin https://github.com/IL_TUO_USERNAME/risiko-rl.git
git push -u origin main
```

Se è la prima volta che usi git da quel PC ti chiederà email/nome:
```powershell
git config --global user.email "tua@email.com"
git config --global user.name "Tuo Nome"
```

## Step 3: Modifica il notebook con il TUO repo

1. Apri `notebooks/train.ipynb` in un editor di testo (anche Notepad)
2. Cerca la riga:
   ```python
   !git clone https://github.com/IL_TUO_USERNAME/il-tuo-repo.git risiko-rl
   ```
3. Sostituiscila con il TUO URL, es:
   ```python
   !git clone https://github.com/EdoMusu1991/risiko-rl.git risiko-rl
   ```
4. Committa e pusha:
   ```powershell
   git add notebooks/train.ipynb
   git commit -m "Aggiorna URL repo nel notebook"
   git push
   ```

## Step 4: Apri il notebook su Colab

1. Vai su [colab.research.google.com](https://colab.research.google.com)
2. **File** → **Apri notebook** → tab **GitHub**
3. Incolla l'URL del tuo repo, premi invio
4. Seleziona `notebooks/train.ipynb`

## Step 5: Abilita la GPU T4

Nel menu di Colab:
1. **Runtime** → **Change runtime type**
2. Hardware accelerator: **T4 GPU**
3. Click **Save**

## Step 6: Esegui il notebook

Esegui le celle in ordine, una alla volta (Shift+Enter):

1. **Cella 1** — Setup Drive + clone repo da GitHub
   - Ti chiede autorizzazione per Drive: accetta
   - Clona il repo nella sessione Colab

2. **Cella 2** — Installa dipendenze
   - `gymnasium`, `sb3-contrib`, `torch`
   - 1-2 minuti

3. **Cella 3** — Verifica simulatore (lancia tutti i test)
   - Devi vedere 136 test passati

4. **Cella 4** — Configura iperparametri training
   - `TOTAL_TIMESTEPS = 1_000_000`
   - `N_ENVS = 8` (8 partite in parallelo)
   - Crea `MaskablePPO` con MlpPolicy

5. **Cella 5** — **TRAINING** (~30-60 min su T4)
   - Salva checkpoint ogni 50k step su Drive
   - Stampa metriche live

6. **Cella 6** — Valutazione su 100 partite
   - Win rate, distribuzione posizioni, durata media

7. **Cella 7** (opzionale) — Carica un checkpoint precedente
   - Per riprendere training o fare valutazioni successive

## Aspettative win rate (baseline random = 25%)

| Step di training | Win rate atteso |
|---|---|
| 100k | 25-30% |
| 500k | 30-40% |
| 1M | **35-50%** ← obiettivo prima sessione |
| 5M | 50-65% |
| 10M+ | 65-80% |

**Importante:** RL è fragile. Se al primo tentativo il bot non impara è normale. Iperparametri da tunare: `learning_rate`, `gamma`, `ent_coef`, `n_steps`. Dopo la prima sessione decideremo cosa cambiare.

## Cosa fare quando il training finisce

1. Scarica l'ultimo checkpoint da Drive (`risiko-rl-checkpoints/risiko_bot_*.zip`)
2. Mandami i numeri di valutazione (cella 6 del notebook)
3. Decidiamo se: continuare training, cambiare iperparametri, o passare a **self-play vero**

## Valutazione locale (offline)

Se vuoi valutare un modello sul tuo PC senza Colab:

```powershell
.venv\Scripts\Activate.ps1
pip install sb3-contrib torch
python scripts\valuta_bot.py path\al\modello.zip --n_partite 100
```

## Troubleshooting

**"out of memory" su Colab:** riduci `N_ENVS` da 8 a 4 nella Cella 4.

**Training lentissimo:** verifica che sia attiva la GPU (Runtime → View resources). Se vedi solo CPU, ripeti Step 5.

**"git clone failed":** se il repo è privato, Colab ti chiede credenziali. Crea un Personal Access Token su GitHub (Settings → Developer settings → Tokens) e usalo come password.

**Test falliscono su Colab ma passano in locale:** apri una issue, mandami output completo della Cella 3.
