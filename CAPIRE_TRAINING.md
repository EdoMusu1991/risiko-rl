# Capire il training del bot RisiKo

Una guida pratica per leggere cosa sta succedendo quando il bot si addestra.

Niente matematica. Esempi reali presi dai training che abbiamo fatto. Quando finirai di leggere, le metriche di Colab ti diranno qualcosa, e capirai perché certe decisioni hanno senso e altre no.

---

## Parte 1 — Come impara il bot

### Reinforcement Learning, in cucina

Immagina di insegnare a un bambino a cucinare la pasta. Hai due strategie:

**Strategia A — Spiegazione esplicita.** Gli dici: "metti l'acqua, accendi il fuoco a 8, aspetta che bolle, butta la pasta, scola dopo 9 minuti". Il bambino esegue gli ordini. Funziona, ma se gli capita una pentola diversa, una pasta diversa, o gli si rompe la fiamma, è in difficoltà.

**Strategia B — Imparare provando.** Lo metti in cucina, lo lasci sperimentare. Lui prova, alla fine assaggia il risultato. Se è buono, sorride. Se è cruda o scotta, fa schifo. Dopo 50 tentativi, ha capito da solo cosa funziona e cosa no — non solo "fai questi passi" ma anche "perché funzionano". Riesce ad adattarsi a pentole nuove.

**Reinforcement Learning è la Strategia B.** Il bot prova mosse, alla fine della partita riceve un voto (vinto, secondo, terzo, ultimo), e impara correlazioni tra "le mosse che ho fatto" e "il voto finale". Dopo migliaia di partite, sa quali decisioni portano a vincere.

### Perché RL invece di programmare le regole?

Programmare un bot che gioca a RisiKo "secondo regole" è impossibile per due motivi:

1. **Le strategie sono troppe.** Non puoi scrivere "se il nemico ha 5 armate al confine attacca, se ne ha 6 difendi". Le combinazioni di territorio, armate, carte, obiettivo sono miliardi. Non puoi enumerarle.

2. **La strategia migliore dipende dagli avversari.** Se i nemici sono aggressivi devi essere difensivo. Se sono passivi devi spingere. Una regola fissa non si adatta.

RL risolve entrambi: il bot impara una "funzione" generica `stato → azione_buona` che si adatta automaticamente.

### Cos'è una "policy"

Il cuore del bot è una funzione chiamata **policy** (o "rete neurale di decisione"). Prende in input lo stato del gioco (le 318→330 feature che gli passiamo) e sputa in output **una distribuzione di probabilità** sulle 1765 azioni possibili.

Esempio: stato corrente = inizio del tuo turno con 5 carri da piazzare in Africa. La policy potrebbe sputare:
- 70% "metti tutti in Egitto"
- 20% "metti 3 in Egitto e 2 in Sahara"
- 8% "spalmali a caso"
- 2% altre

In training (esplorazione) il bot campiona dalla distribuzione, quindi a volte fa anche le mosse meno probabili. In valutazione (`deterministic=True`) sceglie sempre la più probabile.

L'addestramento serve a **modificare quella distribuzione** per spingere verso le azioni che hanno portato a vittorie nelle partite passate.

---

## Parte 2 — PPO in 5 minuti

### Cos'è PPO

**PPO = Proximal Policy Optimization.** È l'algoritmo specifico che stiamo usando per modificare la policy. È stato inventato da OpenAI nel 2017 ed è diventato lo standard per gran parte dell'RL applicato (ChatGPT lo usa, AlphaStar lo usa, mille altri).

Il nome dice cosa fa: **proximal** (vicino), **policy optimization** (ottimizzazione di policy). Cioè: cerca la policy migliore, ma fa **piccoli passi**. Mai cambiamenti drastici.

### Perché "piccoli passi"

Pensa di nuovo al bambino in cucina. Se al primo errore (pasta scotta) gli dici "MAI PIÙ pasta scotta, non farla MAI", lui si traumatizza. La prossima volta tira giù la pasta a 5 minuti per paura, esce cruda. Hai over-correzionato.

L'addestramento di una rete neurale ha lo stesso problema: se a ogni partita persa la rete cambia drasticamente i suoi pesi, oscilla. Non converge mai.

PPO ha un meccanismo che **limita quanto può cambiare la policy ad ogni update**. Tipo un governatore: "vuoi cambiare? Ok, ma di poco. Se vuoi cambiare di più, fallo nel prossimo update". Questo si chiama **clip range** (lo vedrai nelle metriche).

### Actor e Critic

PPO ha due reti neurali, non una:

1. **Actor (la policy)** — quella che decide cosa fare
2. **Critic (la value function)** — quella che stima "quanto vale" la situazione attuale

L'idea: il critic guarda lo stato e dice "da qui ti aspetti di vincere il 60% delle volte" o "da qui sei spacciato, 10%". L'actor usa la stima del critic per capire **se la sua azione ha migliorato o peggiorato la situazione** rispetto a quanto si aspettava.

Esempio: critic dice "vinci il 30% da qui". L'actor fa una mossa. Critic ricalcola: "ora vinci il 50% da qui". L'actor pensa: "ottima mossa, l'aspettativa è salita di 20 punti, rinforzo il comportamento". Se invece fosse scesa al 15%, l'actor avrebbe penalizzato quella mossa.

### Reward sparso vs reward shaping

Nel nostro setup originale, il reward arriva **solo a fine partita**: +1 se vinci, -1 se perdi. Per 300+ step nel mezzo, il bot vede solo zeri.

Questo è "reward sparso" ed è il modo più puro: lasci al bot dedurre cosa è importante. Ma è anche **lentissimo**. Il bot fa migliaia di mosse senza segnale, fatica a correlare cause ed effetti.

Il **reward shaping** dà segnali intermedi piccoli: +0.001 per territorio conquistato, -0.0005 per territorio perso. Non sostituisce il reward terminale (±1 a fine partita) ma lo guida. È come dire al bambino "bene, hai messo l'acqua giusta" invece di aspettare la fine per dire "schifo, non hai messo abbastanza acqua".

Nel tuo training v2 → v3 abbiamo aggiunto reward shaping. È uno dei motivi per cui `ep_rew_mean` è migliorato da -0.547 a -0.272.

---

## Parte 3 — Le metriche, lette davvero

Quando lanci un training, ogni 30 secondi Colab stampa un blocco così:

```
| rollout/                |              |
|    ep_len_mean          | 555          |
|    ep_rew_mean          | -0.0227      |
| time/                   |              |
|    fps                  | 497          |
|    iterations           | 245          |
|    time_elapsed         | 2018         |
|    total_timesteps      | 1003520      |
| train/                  |              |
|    approx_kl            | 0.0215       |
|    clip_fraction        | 0.185        |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.05        |
|    explained_variance   | 0.918        |
|    learning_rate        | 0.0003       |
|    loss                 | -0.0444      |
|    n_updates            | 2440         |
|    policy_gradient_loss | -0.0233      |
|    value_loss           | 0.000889     |
```

Le metriche sono divise in 3 famiglie. Te le spiego con i valori del tuo training reale.

### Famiglia `rollout/` — Come stanno andando le partite

**`ep_rew_mean`** — Reward medio per partita. È **LA metrica chiave**.

Range possibile: tra +1.0 (vince sempre) e -1.0 (perde sempre).

- `-0.547` (training v1, broken): bot dominato, perde quasi sempre
- `-0.272` (training v3 a metà): in miglioramento ma ancora sotto
- `+0.0` o sopra: bot pari/vincente

Trend: deve **salire nel tempo**. Se è piatto o scende, qualcosa non va.

Limite: con reward sparso si muove lentissimo. Cambiamenti reali si vedono ogni 100k-500k step.

**`ep_len_mean`** — Step medi per partita. 

Tradotto: in quanti turni del bot finisce una partita.

- 311 (v1 al 40%): partite veloci. Significa che il bot **muore presto** o vince presto
- 555 (v2 finale): partite più lunghe. Il bot sopravvive di più
- 506 (v3): leggera diminuzione, ok

Cosa cercare: se sale il bot impara a **non farsi eliminare**. Se scende è ambiguo: o vince più velocemente (buono), o muore più velocemente (cattivo). Va incrociato con `ep_rew_mean`.

### Famiglia `time/` — Velocità del training

**`fps`** — Step al secondo. Più alto = training più veloce.

- 497, 507, 659 (i tuoi vari training): valori normali per RisiKo. Il bottleneck è la simulazione Python, non la GPU.

Se cala drammaticamente nel mezzo del training, c'è un problema di memoria o di colli di bottiglia I/O.

**`total_timesteps`** — Step totali processati. Sale linearmente. Quando arriva al `TOTAL_TIMESTEPS` impostato, training finito.

**`iterations`** — Numero di "cicli rollout + update" completati. Dipende da `n_steps`. Con `n_steps=512` ne fai più di con `n_steps=2048`.

**`time_elapsed`** — Secondi dall'inizio. Su Colab T4, 1M step ti porta intorno ai 1500-2000s (25-33 min).

### Famiglia `train/` — Come sta imparando la rete

Queste sono le **metriche tecniche di PPO**. Sono utili a debug.

**`learning_rate`** — Quanto la rete cambia ad ogni update. Costante in genere.

- 0.0003 (3e-4): default, cambiamenti veloci, può essere instabile
- 0.0001 (1e-4): più lento, più stabile, richiede più step

Nel tuo v3 era 1e-4. Per questo serviva più tempo per vedere progresso.

**`entropy_loss`** — Quanto è "incerta" la policy.

Range: di solito da -1.5 (molto incerta, esplora) a 0 (molto sicura, sfrutta).

- -1.39 (v3): bot esplora, prova ancora cose nuove
- -0.857 (v2 finale): bot più "deciso", sfrutta meno

Sale (verso 0) col passare del training: il bot si convince di certe strategie. Se scende troppo presto a -0.2 → bot collassato su una strategia, smette di imparare.

**`explained_variance`** — Quanto bene il critic prevede il reward.

Range: 0 (completamente cieco) a 1 (perfetto).

- 0.91-0.92 (i tuoi training): **eccellente**. Il critic ha capito cosa è una buona posizione.

Se è basso (sotto 0.5) → critic non capisce il gioco, l'actor non riceve guida buona. Brutto segno.

**`approx_kl`** — Quanto la policy nuova si discosta da quella vecchia.

Range: 0.001 (cambiamenti minimi) a 0.1+ (cambiamenti grossi).

- 0.0061-0.022 (i tuoi): range sano. PPO sta facendo i "piccoli passi" come deve.

Se schizza sopra 0.05 → policy sta cambiando troppo, aggressivo. Se sotto 0.001 → quasi non cambia, training stagnante.

**`clip_fraction`** — Quanti aggiornamenti vengono "tagliati" dal clip range di PPO.

Range: 0.0 a 0.3 sano.

- 0.05-0.19 (i tuoi): range buono. Il clip funziona, frena update troppo aggressivi senza tagliare tutto.

Sopra 0.4 → clip troppo stretto, frena troppo. Sotto 0.02 → nessun update problematico (anche bene).

**`value_loss`** — Errore del critic.

- 0.00088-0.002 (i tuoi): bassissimo. Critic ben addestrato.

**`policy_gradient_loss`** — Errore della policy. Negativo è normale per PPO. La magnitudo conta più del segno.

- -0.014, -0.027 (i tuoi): bassi. Policy converge.

**`n_updates`** — Quante volte la rete è stata aggiornata. 

- v2 a 1M: 2440 update
- v3 a 1M: 610 update (4× meno per via di n_steps=2048)

A parità di compute, più update = più apprendimento per dollaro. Ma update troppo piccoli = rumore.

---

## Parte 4 — Come capire se un training "va bene"

Ecco un decalogo pratico. Ogni volta che guardi un blocco di metriche, segui questi step.

### Check 1: il training sta progredendo?

Confronta `ep_rew_mean` tra inizio e adesso:

- Salita visibile (es. da -0.7 a -0.3): training sta funzionando
- Piatto: stagnante. Iperparametri sbagliati, o serve più tempo
- Scende: malato. Probabilmente policy collassata, ent_coef troppo basso

### Check 2: il critic capisce il gioco?

Guarda `explained_variance`:
- Sopra 0.7: critic OK
- 0.4-0.7: critic mediocre, l'actor riceve guida sporca
- Sotto 0.4: critic broken, c'è un problema

### Check 3: il bot esplora ancora?

Guarda `entropy_loss`:
- Sotto -0.5 (verso -1.5): esplora bene
- Tra -0.2 e -0.5: zona di transizione, sta sfruttando
- Sopra -0.2 verso 0: collassato. Smette di imparare cose nuove

Se l'entropy crolla **troppo presto** (es. -0.2 a 100k step), il bot ha trovato una strategia che funziona OK e non vuole cambiarla. Per evitare: aumentare `ent_coef`.

### Check 4: PPO è stabile?

Guarda `clip_fraction` e `approx_kl`:
- `clip_fraction` 0.05-0.3 e `approx_kl` 0.005-0.03: tutto bene
- `clip_fraction` > 0.3: PPO sta tagliando troppo, abbassare `learning_rate`
- `approx_kl` > 0.05: troppi cambiamenti, abbassare `learning_rate`
- Tutto bassissimo: training quasi fermo

### Check 5: i numeri tornano?

Le partite hanno senso? `ep_len_mean` plausibile (300-700 step)? Se vedi `ep_len_mean=10` o `ep_len_mean=10000`, c'è un bug nell'env.

---

## Parte 5 — Cosa è successo nei tuoi training (analisi vera)

Ricapitoliamo i 3 training che hai fatto, con onestà.

### Training v1 (broken bot avversari)

```
ep_rew_mean: -0.547   → bot dominato
ep_len_mean: 377      → partite corte (bot eliminato presto)
explained_variance: 0.91 → critic OK
```

**Diagnosi**: il critic capiva il gioco, l'actor stava imparando, MA il bot era strutturalmente svantaggiato. I 3 avversari giocavano con euristica smart, il bot RL non addestrato giocava completamente random. Asimmetria. Il bot non poteva vincere neanche se avesse imparato perfettamente.

**Win rate finale**: 12% (sotto il random teorico).

### Training v2 (post-fix bot avversari, vecchi iperparametri)

```
ep_rew_mean: -0.0227  → quasi a zero, salto enorme
ep_len_mean: 555      → partite più lunghe, bot non muore presto
explained_variance: 0.92 → ottimo critic
n_updates: 2440       → tante iterazioni
```

**Diagnosi**: il fix dell'asimmetria ha funzionato. Il bot ora compete. `ep_rew_mean` quasi zero significa: vince quanto perde, in media. Buona baseline.

**Win rate finale**: 24%. Esattamente al livello del random teorico simmetrico.

Limitazione: con `learning_rate=3e-4` e `n_steps=512`, il training è "rumoroso". Fa tanti aggiornamenti ma di cattiva qualità.

### Training v3 (nuovi iperparametri, in corso)

```
ep_rew_mean: -0.272 (al 31%)  → in salita
ep_len_mean: 506             → partite stabili
explained_variance: 0.873     → ancora ok
n_updates: 950 (5M step)      → pochi ma di qualità
learning_rate: 0.0001         → lento ma stabile
```

**Diagnosi previsionale**: parametri lenti ma stabili. A 1M step rimane sotto v2 (-0.272 vs -0.027). Ma la traiettoria è in salita. A 5M dovrebbe finire migliore di v2 perché ogni update è di qualità superiore.

**Win rate atteso a 5M**: 30-45%.

Se invece a 5M finisce ancora sotto 30%, sappiamo che PPO da solo non basta. Serve qualcosa di più strutturale (Stage A con opponent embedding, o reward shaping più aggressivo).

---

## Parte 6 — Domande frequenti

### "Il bot ha imparato qualcosa?"

Definizione tecnica: ha imparato se `ep_rew_mean` è significativamente sopra la baseline random.

Per RisiKo simmetrico, baseline random è ~0.0 (1/4 vince, 1/4 quarto, etc, media ≈ 0). Quindi `ep_rew_mean` sopra +0.1 è "ha imparato qualcosa", sopra +0.3 è "ha imparato bene".

Il tuo v2 a -0.027 era **al limite**. v3 deve fare meglio.

### "Perché serve così tanto compute?"

PPO con reward sparso e action space grande è inefficiente per natura. Letteralmente: il bot fa **migliaia** di mosse per ricevere **un singolo bit** di informazione (vinto/perso). Per un gioco semplice come Pong PPO impara in 10 minuti. Per RisiKo serve molto di più.

Riferimenti del mondo reale:
- AlphaStar (StarCraft): **200 anni** di gioco simulato, su un cluster di TPU
- OpenAI Five (Dota): **45.000 anni** di gioco simulato in 10 mesi
- Tu hai 5M step ≈ ~30.000 partite ≈ 0.1 anni di gioco simulato

Siamo dilettanti, fa parte del gioco. Il valore è imparare il metodo, non rivaleggiare con AlphaStar.

### "Quando stop al training? Come capisco se è finito?"

Due segnali:
1. **`ep_rew_mean` plateau** per almeno 1M step. Non sale più.
2. **`explained_variance` saturato** sopra 0.95. Critic non migliora.

Se entrambi, allungare il training non serve. Devi cambiare qualcosa: iperparametri, architettura, reward shaping, observation. Ed è qui che entrano gli stage A/B/C/D che abbiamo discusso.

### "Devo capire la matematica di PPO?"

No. Onestamente. Per usare PPO bene basta:
- Sapere leggere le metriche (questo doc)
- Sapere quali iperparametri sono i più importanti (`learning_rate`, `n_steps`, `ent_coef`, `clip_range`)
- Avere intuizioni sul "se le metriche sono X, prova a cambiare Y"

Il resto lo trovi su Stack Overflow.

---

## Parte 7 — Cosa fare adesso col bot v3

Quando il training finisce e hai i numeri, segui questo flow:

**Se win rate ≥ 35%** → la baseline funziona, passa a Stage A (opponent embedding) sul prossimo training. Il bot ha imparato qualcosa.

**Se win rate 25-35%** → la baseline ha imparato poco. Prima di Stage A, considera:
- Aumentare reward shaping (×3)
- Allungare a 10M step (training più lungo)
- Tornare a `lr=3e-4` ma con `n_steps=1024` (compromesso)

**Se win rate < 25%** → c'è ancora qualcosa di rotto. Stop, debug.

Il visualizzatore di partite (`scripts/visualizza_partita.py`) è il tuo migliore amico in questa fase. Guarda 5 partite vere del bot, capisci dove sbaglia. Spesso il problema è ovvio (es: "non gioca mai i tris" o "attacca sempre il colore sbagliato"), ma lo vedi solo guardando le partite.

---

## Conclusione

PPO è un mestiere. I primi 2-3 training danno risultati mediocri **per chiunque**. Quello che conta è:

1. **Tenere log** di tutti i training (cosa hai cambiato, cosa è successo)
2. **Cambiare una variabile alla volta** (altrimenti non sai cosa ha funzionato)
3. **Misurare con statistiche** (intervalli di confidenza, non singole partite)
4. **Avere pazienza** (un training serio è giorni, non ore)

Adesso, quando vedrai le metriche del training v3, le saprai leggere. Quando ti scriverò di cambiare un iperparametro, capirai perché. Quando una baseline non sale, saprai cosa è normale e cosa no.

Buon training.
