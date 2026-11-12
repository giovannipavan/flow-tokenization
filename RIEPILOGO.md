# Progetto Network Tokenization (CERN IT, Gruppo 1): riepilogo

*Stato al 12 novembre 2026*

## 1. Obiettivo del progetto

Capire se i flussi di rete del CERN (dati **Fireflies / Scitags**) si possono rappresentare come **sequenze di token discreti**, cioè simboli presi da un vocabolario finito, senza perdere l'informazione utile per un compito di predizione.

- **Idea di partenza:** OmniJet-α, in fisica delle alte energie, trasforma i jet di particelle in token discreti e ci addestra modelli in stile LLM. Ci chiediamo se la stessa idea funziona sui dati di rete.
- **Due gruppi, stessi dati e stesso split:**
  - Gruppo 1 (noi): **tokenizzazione**;
  - Gruppo 2: **JEPA**, cioè rappresentazioni continue.
- **Task a valle (dal README):** predire `delta_rtt = end_rtt − start_rtt`, cioè **quanto cambia l'RTT durante il trasferimento**. Va confermato con Carmen, perché le slide parlano di "RTT prediction".
- **Tre strategie di tokenizzazione da confrontare:**
  - **feature-level:** un token per campo, es. `DST_PORT_1094`;
  - **flow-level:** un token per combinazione di campi, es. `TCP_DST1094`;
  - **learned / VQ** (opzionale): codebook imparato.
- **Deliverable:** codice riproducibile, report scientifico breve, presentazione al CERN.

## 2. Contesto: cosa sono i Fireflies

- I **fireflies** sono piccoli pacchetti **UDP** in formato syslog/JSON. Lo storage che esegue un trasferimento (es. XRootD) li invia accanto al flusso vero.
- Servono a dire alla rete **chi** sta trasferendo e **perché**: esperimento (es. CMS) e attività. È il meccanismo di *flow labelling* del framework Scitags.
- Ogni flusso genera un evento di **start** e uno di **end**, con la 5-tupla (IP e porte di sorgente e destinazione, protocollo) e le statistiche TCP del kernel, incluso l'**RTT**.
- **Concetti chiave:**
  - **RTT:** tempo di andata e ritorno di un pacchetto;
  - **sito CRIC:** il nome di un centro di calcolo WLCG nel catalogo CRIC (es. `CERN-PROD`);
  - **ASN:** il numero dell'operatore di rete (es. 513 = CERN);
  - **prefisso IP:** la parte "di rete" di un indirizzo (`/24` in IPv4, `/48` in IPv6).

## 3. Il dataset

- `flows.parquet`: **67.244.519 flussi**, 23 colonne, letti con **DuckDB** senza caricarli in RAM.
- **Ogni riga è un flusso completo**, cioè start ed end già accoppiati tramite `(flow_id, flow_start)`.
- **Periodo coperto:** 8–14 ottobre 2026 (UTC).
- **Split cronologico già fornito (70/15/15), basato su `flow_start`:**

| Split | Periodo (UTC) | Righe originali |
|---|---|---|
| train | 8 ott 00:00 → 12 ott 17:12 | 47.071.163 |
| validation | 12 ott 17:12 → 13 ott 22:18 | 10.086.678 |
| test | 13 ott 22:18 → 14 ott 23:59 | 10.086.678 |

**Regole anti-leakage (dal README):**
- **Non si possono usare come input:** `end_rtt`, `delta_rtt`, `flow_end`, la durata del flusso, `endpoints_match`, `endpoints_reversed`, `start_fields_match`, e `flow_id` come predittore.
- `start_rtt` è ammesso, perché la predizione avviene dopo l'evento di start.
- Siti e ASN CRIC sono un arricchimento esterno: va verificato che siano disponibili al momento della predizione.
- Vocabolari, scaler, soglie e imputazioni vanno costruiti **solo sul train**.

## 4. Cosa abbiamo visto

### 4.1 Prime righe
- **Unità dell'RTT:** con ogni probabilità **millisecondi**. I valori tornano con le distanze: ~0,1 ms all'interno dello stesso sito, ~11 ms Karlsruhe–CERN, ~95 ms Lione–MIT, ~236 ms Taiwan–Nebraska. La risoluzione minima è 0,002 ms.
- IPv4 e IPv6 compaiono mescolati.
- Le porte di destinazione sono quasi sempre **1094/1095**, cioè XRootD.
- `meta.application` ha valori molto eterogenei (`cmsRun`, `python3.10`, `unknown`…).

### 4.2 Valori mancanti
I campi principali (IP, porte, timestamp, RTT, protocollo, versione IP, esperimento, attività, hostname) **non hanno NULL** né stringhe vuote.

| Campo | NULL | % | Note |
|---|---|---|---|
| sito / ASN sorgente | 12.339.289 | 18,35% | mancano sempre insieme |
| sito / ASN destinazione | 2.707.283 | 4,03% | mancano sempre insieme |
| `meta.application` | 475.477 | 0,71% | più 5.541.029 `unknown` (8,24%) |
| `meta.experiment_name` | 0 | 0% | 289.901 `unknown` (0,43%) |

**Combinazioni:** entrambi i lati noti 78,4%, solo sorgente ignota 17,6%, solo destinazione ignota 3,3%, entrambi ignoti 0,8%.

**Interpretazione:** se un IP non è nel catalogo CRIC, sia il sito sia l'ASN restano vuoti. Le sorgenti, cioè client e worker node, sono censite peggio delle destinazioni, che sono soprattutto server di storage.

### 4.3 Distribuzione degli RTT

| | min | p01 | p25 | p50 | p75 | p99 | max | n < 0 | n = 0 |
|---|---|---|---|---|---|---|---|---|---|
| `start_rtt` | 0 | 0,12 | 9,85 | 17,8 | 46,5 | 249 | 32.182 | 0 | 773 |
| `end_rtt` | 0 | 0,16 | 14,6 | 21,8 | 47,6 | 250 | 120.322 | 0 | 357.462 |
| `delta_rtt` | −14.994 | −9,0 | 0,06 | 4,44 | 5,25 | 15,5 | 120.017 | 9,68 M | 242.527 |

- Gli RTT non sono mai negativi, ma ci sono degli **zeri**, soprattutto in `end_rtt`.
- I **`delta_rtt` negativi sono legittimi:** indicano che l'RTT è sceso durante il trasferimento.
- `delta_rtt` ha **code pesanti**: la deviazione standard è 113 contro una mediana di 4,4.

### 4.4 Il target è bimodale
- L'istogramma di `delta_rtt` mostra **due picchi**: uno stretto a **0** e uno a **~5 ms**.
- **Prima osservazione:** sembrava che dipendesse da `endpoints_reversed`. Mediana 4,95 ms per i flussi `reversed` (66%) contro 0,06 ms per quelli `match` (34%).
- **Controllando per la durata, la differenza si riduce molto.** Il picco a ~5 ms riguarda soprattutto i **trasferimenti lunghi**, sopra il minuto, ed è presente in entrambi i gruppi. I flussi `reversed` sono quasi tutti lunghi, quelli `match` soprattutto brevi.
  - È un caso di **confondimento**: la durata influenza sia il flag sia il target.
  - A parità di durata resta comunque una differenza tra i gruppi.
- **Ipotesi:** il valore di ~5 ms è costante e indipendente dalla distanza. È compatibile con il **ritardo di coda obiettivo di fq_codel**, che vale 5 ms di default in Linux. Sarebbe quindi un fenomeno reale di congestione, non un errore di misura. Da verificare con il CERN.
- **Conseguenza per il modello:** la durata è il predittore più forte, ma è **vietata**. Il modello dovrà inferire il "tipo di trasferimento" dai campi disponibili allo start, cioè applicazione, attività, siti, porte e ora.

### 4.5 Gli RTT pari a zero sono misure mancanti
- **`end_rtt = 0` (357.462 righe):**
  - il 99,99% di questi flussi ha `start_rtt ≥ 1 ms`, quindi uno zero finale è fisicamente impossibile;
  - circa il **90%** riguarda flussi verso **USCMS-FNAL-WC1** (Fermilab). Per alcune coppie la quota arriva al 100% (UCSDT2 → FNAL 100%, Nebraska → FNAL 95%, MIT → FNAL 89%), mentre per coppie frequenti come RAL → CERN è ≤ 0,02%;
  - è quindi un **problema del produttore di telemetria su FNAL**, non un comportamento della rete.
- **`start_rtt = 0` (773 righe):** sparsi su coppie diverse e non legati a FNAL. Il loro RTT finale è normale, quindi anche questi zeri sono misure mancate.
- Il problema **cala nel tempo**: righe rimosse 0,72% nel train, 0,17% nella validation, 0,003% nel test.

## 5. Cosa abbiamo deciso

### 5.1 Pulizia applicata
È stata creata la vista **`flows_clean`**:
- filtro `start_rtt > 0 AND end_rtt > 0`;
- colonna `t_start` (timestamp convertito);
- colonna `split` secondo le regole di `splits.json`.

| Split | Originali | Dopo la pulizia | Rimosse | % |
|---|---|---|---|---|
| train | 47.071.163 | 46.730.686 | 340.477 | 0,72% |
| val | 10.086.678 | 10.069.254 | 17.424 | 0,17% |
| test | 10.086.678 | 10.086.352 | 326 | 0,003% |
| **totale** | **67.244.519** | **66.886.292** | **358.227** | **0,53%** |

**Perché non è leakage:** escludiamo righe con target **non valido**, non righe scomode. Il filtro vale in tutti gli split, e i flussi verso FNAL con misure valide restano nel dataset.

### 5.2 Cosa teniamo
- I `delta_rtt` **negativi**, perché sono reali.
- Gli **outlier estremi**, per ora. L'eventuale taglio si deciderà in fase di modellazione, con soglie calcolate sul train.
- NULL e `unknown` di `meta.application` come **due categorie distinte**.

### 5.3 Rappresentazione della "posizione" (sito / prefisso)
Non è ancora stata applicata nessuna imputazione. La proposta è questa:
- **Rappresentazione principale: prefisso IP + sito CRIC, insieme**, cioè due token per lato:
  ```
  SRC_PFX_2001:630:10::/48   SRC_SITE_MISSING
  DST_PFX_2001:630:10::/48   DST_SITE_UKI-LT2-Brunel
  ```
  Il prefisso c'è sempre, il sito aggiunge un'informazione condivisa e leggibile, e il modello impara a combinarli.
- **Variante B, da valutare:** ricostruire il sito mancante dal prefisso IP. La tabella prefisso → sito si costruisce **solo sul train**, e si usa solo per i prefissi con purezza ≥ 95% e supporto ≥ 100, aggiungendo un flag di imputazione.
- **Niente imputazione stocastica:** aggiungerebbe solo rumore a un problema di predizione.

### 5.4 Metriche
Si riportano MAE e RMSE, privilegiando il **MAE**: con code così pesanti, l'RMSE sarebbe dominato da pochi outlier. A queste si aggiungono la dimensione del vocabolario, la memoria e la percentuale di token `UNK`/`RARE` in validation e test.

## 6. Domande aperte per Carmen / CERN
1. Il picco a ~5 ms nei trasferimenti lunghi è un fenomeno reale di coda (es. fq_codel) o una differenza tra gli strumenti che misurano start ed end?
2. Gli `end_rtt = 0` concentrati su USCMS-FNAL-WC1 sono un problema noto del produttore? È stato corretto durante la settimana?
3. Il target è `delta_rtt` (README) o l'RTT (slide)?
4. I campi CRIC (sito, ASN) sono disponibili in tempo reale, cioè possiamo usarli come input?
5. *(Idea extra)* Interessa un task secondario di **qualità della telemetria**, cioè predire quando la misura di end fallirà?

## 7. Prossimi step

### Esplorazione (in corso)
1. **Cardinalità delle variabili categoriche sul train** (`approx_count_distinct`): IP, porte, siti, ASN, applicazioni, hostname, esperimenti, attività. Determina la dimensione dei vocabolari.
2. **Concentrazione dei valori:** per i campi grandi, quanta parte dei flussi coprono i top-k valori. Serve a scegliere le soglie di frequenza e il bucket `OTHER`/`RARE`.
3. **Prefissi IP:** calcolo dei prefissi `/24` e `/48` con il modulo `ipaddress`, sugli IP unici. Si contano i prefissi distinti e si misura **quante sorgenti senza sito hanno un prefisso noto**: è il numero che decide se vale la pena fare la variante B.
4. **Sovrapposizione train/test:** quanti siti, prefissi e hostname del test compaiono anche nel train, cioè quanto sarà frequente `UNK`.
5. **Andamento nel tempo:** flussi per ora e `delta_rtt` medio per ora, per vedere se ci sono cicli giornalieri ed eventuali feature temporali.
6. **Relazione tra `start_rtt` e `delta_rtt`**, e tra `delta_rtt` e le variabili categoriche principali (applicazione, attività, coppia di siti).
7. *(Facoltativo)* Flussi che attraversano i confini degli split (start nel train, end dopo il cutoff): decidere se escluderli dal train.

### Modellazione
8. **Baseline senza tokenizzazione:**
   - mediana di `delta_rtt` globale e per coppia di siti/prefissi, calcolata sul train;
   - gradient boosting su `start_rtt` + campi categorici codificati in modo standard.
9. **Tokenizzazione feature-level:** vocabolari costruiti sul train, embedding → concatenazione → MLP → `delta_rtt`.
10. **Tokenizzazione flow-level** ed eventualmente **VQ**.
11. **Ablation:** sito vs prefisso vs entrambi, granularità del prefisso (`/16`, `/24`, `/48`, `/64`), `start_rtt` incluso o escluso, campi rimossi uno alla volta.
12. **Valutazione finale sul test**, inclusa l'analisi separata dei flussi verso FNAL.

### Report
13. Mantenere aggiornata la sezione "Pulizia dei dati" del notebook e aggiungere le sezioni "Variabili categoriche" e "Prefissi IP" man mano.