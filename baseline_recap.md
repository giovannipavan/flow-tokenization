# Baseline per la predizione di `delta_rtt` – Recap del lavoro (notebook `03_baseline`)

*Progetto: tokenizzazione dei network flow CERN (dati Scitags Firefly), Gruppo 1, con Carmen Misa (CERN IT). Stato al 16 novembre 2026.*

---

## 0. In una pagina

- **Obiettivo delle baseline:** fissare un *pavimento* (quanto si sbaglia senza informazione) e un *soffitto pratico* (quanto si ottiene con feature grezze e modelli forti), per poter dire poi quanta informazione una tokenizzazione conserva. La predizione di `delta_rtt = end_rtt − start_rtt` è il banco di prova, non il fine.
- **Fatto finora:** ambiente e dati sistemati; `flows_prepared.parquet` rigenerato e verificato; funzione di valutazione unica con metriche robuste; **B0** (mediana globale) e **B1** (mediana per tratta, 6 varianti) completate e analizzate.
- **Risultati principali (validation):**

| Modello | MAE | RMSE | RMSE robusto † | P90 \|err\| | P99 \|err\| | mediane memorizzate |
|---|---|---|---|---|---|---|
| B0 mediana globale | 3,298 | 47,289 | 5,812 | 4,510 | 23,552 | 1 |
| B1 prefissi L2, server → client (migliore) | **2,209** | 47,233 | 5,377 | 4,861 | 23,068 | 2.166 |

  Valori in ms. † RMSE calcolato solo sui flow con `delta_rtt` tra i quantili 0,05 % e 99,95 % del train, cioè [−184,23, 58,82] ms (insieme fisso, uguale per tutti i modelli).

- **Scoperte chiave:**
  1. Lo **0,1 % dei flow** (~10.000) produce il **98,8 % dell'errore quadratico**: l'RMSE completo è deciso dagli estremi, da cui l'RMSE robusto †.
  2. La **tratta** (coppia di estremi) riduce la MAE del 33 %; conta il **ruolo server/client**, non il verso di registrazione; i **prefissi IP L2** battono i siti CRIC.
  3. Dentro ogni tratta restano **due regimi** (flow brevi ~0 ms, lunghi ~5 ms): la MAE scende ma il P90 sale.
  4. **Anomalia FNAL:** sui flow con USCMS-FNAL-WC1 come client la B1 peggiora (MAE 1,64 → 3,40). La mediana di `delta_rtt` di questi flow sembra legata all'attivazione del guasto di telemetria `end_rtt = 0`. Causa ancora aperta.

---

## 1. Contesto minimo

### Dati
- `flows.parquet`: 67.244.519 flow (una riga = un flow TCP ricostruito accoppiando il Firefly di *start* e quello di *end*), 23 colonne, inizio dei flow tra il 7 ottobre 23:59 e il 14 ottobre 23:59 UTC.
- **Target:** `delta_rtt` in ms. Bimodale (picco a 0 e a ~5 ms, quest'ultimo legato ai flow lunghi), 14,4 % negativi, code estreme (da −14.994 a +120.017 ms sull'intero dataset).
- **Input vietati** (richiedono l'evento di fine → data leakage): `end_rtt`, `delta_rtt`, `flow_end`, durata, `endpoints_match`, `endpoints_reversed`, `start_fields_match`. `flow_id` solo come identificatore. `start_rtt` ammesso (si prevede dopo l'evento di start).

### Split cronologico (70/15/15 su `flow_start`, confini in `splits.json`)

```
8 ott ───────── train ───────── 12 ott 17:12 │ validation │ 13 ott 22:18 │ test │ 14 ott
```

| Split | Prima della pulizia | Dopo la pulizia (`start_rtt > 0 AND end_rtt > 0`) |
|---|---|---|
| train | 47.071.163 | 46.730.686 |
| validation | 10.086.678 | 10.069.254 |
| test | 10.086.678 | 10.086.352 |

- **Train:** si stima tutto (mediane, soglie, vocabolari, pesi).
- **Validation:** si confrontano modelli e varianti, si scelgono iperparametri (usata molte volte).
- **Test:** usato **una sola volta**, alla fine.

### Perché non k-fold
1. **Tempo:** fold casuali farebbero usare il futuro per prevedere il passato, e flow vicini nel tempo (stessa congestione, stesso trasferimento spezzato) finirebbero in train e validation → errore ottimista (leakage). Lo split cronologico simula l'uso reale ed è condiviso con il Gruppo 2.
2. **Quantità di dati:** con 10 M flow in validation la stima dell'errore è già molto precisa.
3. **Costo:** k addestramenti senza beneficio reale.
- Alternativa futura, se arrivano più dati: cross-validation *temporale* (rolling / forward-chaining).

### Uso del test durante l'esplorazione (trasparenza)
- Sul test erano stati calcolati solo conteggi e statistiche descrittive (righe, siti mancanti, tasso UNK, incidenza `end_rtt = 0`). Nessuna scelta di modellazione ne dipende.
- **Unica decisione presa guardando il test:** non normalizzare `meta.application`. È stata **ri-verificata sulla sola validation** (§5) ed è confermata.
- **Da ora il test è congelato:** tutte le query usano `split IN ('train', 'val')`.

---

## 2. Ambiente e problemi risolti

| Problema | Causa | Soluzione |
|---|---|---|
| `ArgumentError: Package json not found` | notebook aperto con kernel **Julia** | selezionare un kernel Python |
| `ModuleNotFoundError: duckdb` | ambiente Python senza il pacchetto | `%pip install duckdb pyarrow pandas numpy` (installa nel kernel in uso) e riavvio del kernel |
| `No files found that match the pattern "data/flows.parquet"` | percorso relativo errato | dati in `flow-tokenization/data/student_dataset/`; notebook in `results/` → `PROJECT_ROOT = Path.cwd().parent` |
| `Relations no longer have fetchone()` | DuckDB 2.0: `con.sql(...)` è una relation *lazy* | usare `con.execute(...).fetchone()` / `.df()` |
| `flows_prepared.parquet` assente | il file era stato generato su un altro PC | notebook autonomo `genera_flows_prepared.ipynb` (§3) |
| `Combining QUALIFY with GROUP BY ALL is not supported` | limite di DuckDB | scrivere esplicitamente le colonne in `GROUP BY` |

**Versioni (da riportare nel report):** Python 3.14.0, pandas 3.1.1, numpy 2.3.5, DuckDB 2.0.0.

**Convenzione DuckDB:** `con.execute(...)` per tutto ciò che va portato in Python (numeri, DataFrame); `con.sql(...)` solo per anteprime.

**Bug latente corretto:** con `SET preserve_insertion_order = false` DuckDB non garantisce l'ordine delle righe di un `UNION ALL`. La vecchia `valuta` leggeva la riga globale con `.loc[0]` → ora le righe sono ordinate esplicitamente e la riga globale si legge **per nome**.

---

## 3. Rigenerazione di `flows_prepared.parquet`

Notebook autonomo (`genera_flows_prepared.ipynb`) con i soli passaggi necessari:
1. vista `flows_clean` (pulizia + split);
2. tabella IP → prefissi (L1 /16·/32, **L2 /24·/48 default**, L3 /24·/64) con il modulo `ipaddress`, IPv4-mapped convertiti;
3. porte di servizio stimate **sul train** (porte < 32768 con ≥ 1.000 flow; 42 nominate + intervalli 20000–24999 e 30000–30999) e macro `servizio`, `porta_cat`, `porta_fascia`;
4. vista `flows_prepared` (orientamento `C2S`/`S2C` con regola a cascata, colonne canoniche `server_*`/`client_*`) e scrittura in Parquet ZSTD; interruttore `SOVRASCRIVI = False` per non riscrivere 5 GB per errore;
5. verifica.

**Verifica superata:** righe per split identiche, 41 colonne, nessun NULL, 0 errori nelle colonne canoniche, `pct_S2C` 17,773, `pct_incerto` 0,266, 0 durate negative, 5,29 GB.

---

## 4. Impianto di valutazione

### Setup del notebook

```python
import json, time, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd

PROJECT_ROOT = Path.cwd().parent
DATA_DIR = PROJECT_ROOT / "data" / "student_dataset"
FLOWS, SPLITS = DATA_DIR / "flows.parquet", DATA_DIR / "splits.json"
SEED = 42
RESULTS = []

con = duckdb.connect()
con.execute("SET threads TO 8")
con.execute("SET memory_limit = '8GB'")
con.execute(f"SET temp_directory = '{PROJECT_ROOT / 'duckdb_tmp'}'")
con.execute("SET preserve_insertion_order = false")
con.execute("SET TimeZone = 'UTC'")

PREPARED = DATA_DIR / "flows_prepared.parquet"
con.execute(f"CREATE OR REPLACE VIEW fp AS SELECT * FROM read_parquet('{PREPARED}')")
```

### Metriche: cosa misurano e perché

| Metrica | Definizione | Costante ottima | Lettura |
|---|---|---|---|
| **MAE** (principale) | media di \|errore\| | mediana | errore tipico; ogni ms pesa uguale |
| **RMSE** | √(media di errore²) | media | dominato dagli errori grandi |
| **RMSE robusto †** | RMSE sui soli flow *tipici* | — | "quanto sbaglio in grande sui flow normali" |
| **P90 / P99 \|err\|** | 90° / 99° percentile dell'errore assoluto | — | "nel 90 % dei flow sbaglio meno di X ms" |

- **Rapporto RMSE/MAE come indicatore di code pesanti:** ≈ 1,25 per errori gaussiani, ≈ 1,41 per Laplace; **14** per la nostra B0. Esempio: 1000 errori da 1 ms e uno da 1000 ms → MAE ≈ 2, RMSE ≈ 31,6.
- **Coerenza loss–metrica:** si valuta con la MAE, quindi si addestra con loss L1 (LightGBM `objective = "l1"`, MLP con L1 o Huber). Con l'MSE il modello stimerebbe la media condizionata invece della mediana condizionata.

### Perché un RMSE robusto, e come definirlo
- **Diagnostica sulla B0:** lo 0,1 % dei flow peggiori produce il **98,8 %** dell'errore quadratico (l'1 % il 99,4 %).
- **Modo sbagliato:** togliere, per ogni modello, lo 0,1 % dei *suoi* errori peggiori → ogni modello valutato su flow diversi, e premiato per i suoi errori più gravi.
- **Modo corretto (adottato):** un insieme **fisso** di flow tipici, definito sul **valore vero** con soglie del **train**:
  - quantili 0,05 % e 99,95 % del train → **LO = −184,23 ms, HI = +58,82 ms** (asimmetriche perché seguono la forma della distribuzione);
  - tipici: 99,900 % del train (per costruzione), **99,918 %** della validation (8.224 flow esclusi).
- **Regola di presentazione:** ogni tabella riporta **sempre entrambi** gli RMSE; quello robusto è marcato con **†** e accompagnato dalla nota sulla definizione.
- La validation ha ~18 % di valori estremi in meno del train (8.224 contro ~10.069 ± 100 attesi): differenza reale. Va nella stessa direzione dei cambiamenti osservati nella settimana (calo degli `end_rtt = 0` e del traffico da reti senza sito CRIC), ma il legame causale **non è verificato** (le RTT nulle sono già state rimosse).

```python
Q_LO, Q_HI = 0.0005, 0.9995
LO, HI = con.execute(f"""
    SELECT quantile_cont(delta_rtt, [{Q_LO}, {Q_HI}]) FROM fp WHERE split = 'train'
""").fetchone()[0]
NOTA_ROBUSTO = (f"† RMSE robusto: calcolato solo sui flow con delta_rtt in [{LO:.2f}, {HI:.2f}] ms, "
                f"cioè tra i quantili {Q_LO:.2%} e {Q_HI:.2%} del train (insieme fisso, uguale per tutti i modelli).")
```

### Sottogruppi di analisi

| Sottogruppo | Definizione | Perché |
|---|---|---|
| globale | tutte le righe | metrica ufficiale |
| FNAL | server o client = `USCMS-FNAL-WC1` | copertura ridotta dopo la rimozione degli `end_rtt = 0` |
| sito mancante | server o client = `MISSING` | ~18 % dei flow senza sito CRIC |
| verso incerto | `verso_incerto = TRUE` | orientamento server/client ambiguo (0,27 %) |
| flow > 1 min | `an_durata_s > 60` | origine del picco a ~5 ms; durata usata **solo per filtrare** |

### Funzione di valutazione

```python
SOTTOGRUPPI = {
    "globale":       "TRUE",
    "FNAL":          "server_site = 'USCMS-FNAL-WC1' OR client_site = 'USCMS-FNAL-WC1'",
    "sito mancante": "server_site = 'MISSING' OR client_site = 'MISSING'",
    "verso incerto": "verso_incerto",
    "flow > 1 min":  "an_durata_s > 60",
}

def valuta(vista):
    """vista: vista DuckDB con le colonne di fp più 'pred' (solo righe di validation)."""
    parti = []
    for ordine, (nome, cond) in enumerate(SOTTOGRUPPI.items()):
        parti.append(f"""
            SELECT {ordine} AS ordine, '{nome}' AS gruppo, COUNT(*) AS n,
                   AVG(ABS(delta_rtt - pred))                         AS mae,
                   SQRT(AVG(POWER(delta_rtt - pred, 2)))              AS rmse,
                   SQRT(AVG(POWER(delta_rtt - pred, 2))
                        FILTER (WHERE delta_rtt BETWEEN {LO} AND {HI})) AS rmse_robusto,
                   quantile_cont(ABS(delta_rtt - pred), 0.90)         AS p90_abs,
                   quantile_cont(ABS(delta_rtt - pred), 0.99)         AS p99_abs
            FROM {vista} WHERE {cond}
        """)
    query = "SELECT * FROM (" + " UNION ALL ".join(parti) + ") ORDER BY ordine"
    return con.execute(query).df().drop(columns="ordine")

NOMI_METRICHE = {"mae": "MAE", "rmse": "RMSE", "rmse_robusto": "RMSE robusto †",
                 "p90_abs": "P90 |err|", "p99_abs": "P99 |err|"}

def mostra(ris):
    display(ris.rename(columns=NOMI_METRICHE).round(3))
    print(NOTA_ROBUSTO)
```

**Tabella finale (`RESULTS`):** modello, MAE, RMSE, RMSE robusto †, P90, P99, parametri (o mediane memorizzate), **tempo di solo training** (la valutazione è esclusa dal cronometro). Rieseguire una cella sostituisce la riga del modello, non la duplica.

---

## 5. Ri-verifica della scelta su `application` (solo validation)

- In validation compaiono **15 valori** di `application` assenti dal train, per **3.694 righe (0,037 %)**.
- **Nessuno** è una versione di Python: sono tutti job di analisi (configurazioni dell'Higgs come `vbfhmm_config_…` e `vhmm_config_…`, task O2 di ALICE come `o2-analysis-…`, eseguibili generici come `main.exe`).
- Nessuna normalizzazione li ricondurrebbe a valori noti → la scelta di **non normalizzare** è confermata senza usare il test; i valori nuovi diventano `UNK`.

---

## 6. Baseline 0 – Mediana globale

- **Idea:** prevedere sempre la mediana di `delta_rtt` del train. È la costante che minimizza la MAE.
- **Calcolata sul train completo** (46,7 M righe) in DuckDB, senza campionamento: nessun rumore di stima. Tempo: 1,4 s.
- **Mediana 4,398 ms, media 3,172 ms.** La mediana cade nel picco a ~5 ms; la media è trascinata in basso dal picco a 0 e dai negativi.

| Gruppo | n | MAE | RMSE | RMSE robusto † | P90 \|err\| | P99 \|err\| |
|---|---|---|---|---|---|---|
| globale | 10.069.254 | 3,298 | 47,289 | 5,812 | 4,510 | 23,552 |
| FNAL | 215.471 | 1,636 | 3,193 | 2,666 | 4,381 | 6,279 |
| sito mancante | 2.122.477 | 3,539 | 16,574 | 6,885 | 4,847 | 25,834 |
| verso incerto | 32.662 | 4,570 | 8,801 | 4,790 | 4,423 | 5,541 |
| flow > 1 min | 6.632.723 | 3,129 | 58,019 | 6,453 | 4,643 | 27,535 |

**Diagnostica:**
- Prevedere la **media** invece della mediana: MAE 3,388 (+2,7 %) → conferma teorica.
- **Nessun drift del centro:** mediana della validation 4,442 contro 4,398 del train.
- Quantili di `delta_rtt` in validation: p0,1 = −63,1; p1 = −4,54; p5 = −0,23; p25 = 0,06; p50 = 4,44; min −2.108, max +32.518 ms.
- **P90 ≈ 4,5 ≈ mediana prevista:** i flow del picco a 0 sbagliano di circa la distanza tra i due picchi. Il margine di miglioramento principale è distinguere i due regimi.
- **Flow > 1 min:** 66 % della validation; MAE più bassa ma code più pesanti.

---

## 7. Baseline 1 – Mediana per tratta con fallback gerarchico

### Idea
Una mediana del train per ogni **tratta** (coppia di estremi); per ogni flow della validation si prevede la mediana della sua tratta. È una *lookup table* senza parametri ottimizzati: misura quanto spiega il solo "dove". È anche un **token flow-level rudimentale** (una combinazione di campi → un valore), con i problemi tipici: combinazioni rare, vocabolario che cresce.

### Identificatori della posizione di un estremo

| Livello | Colonne | Cosa identifica | Valori distinti (train) | Sempre presente? |
|---|---|---|---|---|
| IP | `src_ip`, `dst_ip` | una macchina | ~27.000 | sì |
| prefisso L2 | `*_L2` | /24 IPv4, /48 IPv6: la rete di un istituto | 720 | sì |
| sito CRIC | `*_site` | centro di calcolo WLCG | 90 | no (`MISSING` nel ~18 %) |
| ASN | `*_asn` | operatore di rete | 64 | come il sito |

### Varianti: 2 identificatori × 3 orientamenti

| | non orientata {A, B} | server → client | src → dst |
|---|---|---|---|
| siti CRIC | B1 site non_orientata | B1 site server_client | B1 site src_dst |
| prefissi L2 | B1 L2 non_orientata | B1 L2 server_client | B1 L2 src_dst |

- **Non orientata:** conta solo la tratta; (A, B) e (B, A) sono lo stesso gruppo (chiave `LEAST`, `GREATEST`).
- **Server → client:** conta quale estremo offre il servizio (ruolo).
- **Src → dst:** conta il verso con cui il Firefly ha registrato il flow (XRootD client → server, HTTP/WebDAV server → client).
- **Avvertenza:** il verso dei byte non è osservato (il campo `usage` del Firefly non è nel dataset); l'"ordine" riguarda ruoli o registrazione.

Esempio (CERN e RAL, quattro flow): la variante non orientata mette tutti i flow in un gruppo; server → client li divide per chi fa da server; src → dst per come sono stati registrati.

### Fallback (uguale per tutte le varianti, per un confronto pulito)
coppia (siti o prefissi, con l'orientamento della variante) → coppia di **ASN** (stesso orientamento) → **mediana globale**. Un gruppo si usa solo se ha almeno **N = 1.000** righe di train (stessa soglia dei vocabolari; compromesso bias–varianza: gruppi piccoli sono specifici ma rumorosi).

### Codice

```python
ORIENTAMENTI = {
    "non_orientata": ("LEAST({p}server_{x}, {p}client_{x})", "GREATEST({p}server_{x}, {p}client_{x})"),
    "server_client": ("{p}server_{x}", "{p}client_{x}"),
    "src_dst":       ("{p}src_{x}",    "{p}dst_{x}"),
}

def chiavi(orient, x, p=""):
    k1, k2 = ORIENTAMENTI[orient]
    return k1.format(p=p, x=x), k2.format(p=p, x=x)

def baseline1(ident, orient, N=1000):
    t0 = time.time()
    for livello, x in [(1, ident), (2, "asn")]:
        k1, k2 = chiavi(orient, x)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE med_l{livello} AS
            SELECT {k1} AS k1, {k2} AS k2, MEDIAN(delta_rtt) AS med, COUNT(*) AS n
            FROM fp WHERE split = 'train'
            GROUP BY ALL HAVING COUNT(*) >= {N}
        """)
    t_train = time.time() - t0

    a1, b1 = chiavi(orient, ident, p="f.")
    a2, b2 = chiavi(orient, "asn", p="f.")
    con.execute(f"""
        CREATE OR REPLACE VIEW pred_b1 AS
        SELECT f.*,
               COALESCE(m1.med, m2.med, {mediana}) AS pred,
               CASE WHEN m1.med IS NOT NULL THEN '1 coppia {ident}'
                    WHEN m2.med IS NOT NULL THEN '2 coppia ASN'
                    ELSE '3 globale' END AS livello
        FROM fp AS f
        LEFT JOIN med_l1 AS m1 ON {a1} = m1.k1 AND {b1} = m1.k2
        LEFT JOIN med_l2 AS m2 ON {a2} = m2.k1 AND {b2} = m2.k2
        WHERE f.split = 'val'
    """)
    ris = valuta("pred_b1")
    livelli = con.execute("""
        SELECT livello, COUNT(*) AS n,
               100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS pct,
               AVG(ABS(delta_rtt - pred)) AS mae
        FROM pred_b1 GROUP BY livello ORDER BY livello
    """).df()
    n_param = sum(con.execute(f"SELECT COUNT(*) FROM med_l{l}").fetchone()[0] for l in (1, 2)) + 1
    return ris, livelli, n_param, t_train
```

Concetti SQL usati: `LEAST`/`GREATEST` per la chiave non orientata; `TEMP TABLE` per materializzare le mediane; `HAVING` per la soglia sui gruppi; `LEFT JOIN` + `COALESCE` per il fallback in una riga; window function `SUM(COUNT(*)) OVER ()` per le percentuali.

### Risultati globali (validation)

| Variante | MAE | RMSE | RMSE robusto † | P90 \|err\| | P99 \|err\| | mediane | tempo (s) |
|---|---|---|---|---|---|---|---|
| B0 mediana globale | 3,298 | 47,289 | 5,812 | 4,510 | 23,552 | 1 | 1,4 |
| B1 siti, non orientata | 2,463 | 47,252 | 5,515 | 4,977 | 23,283 | 816 | 1,9 |
| B1 siti, server → client | 2,270 | 47,241 | 5,419 | 4,909 | 23,188 | 1.077 | 1,3 |
| B1 siti, src → dst | 2,401 | 47,248 | 5,480 | 4,953 | 23,276 | 1.029 | 1,2 |
| B1 L2, non orientata | 2,338 | 47,241 | 5,440 | 4,884 | 23,240 | 1.920 | 2,3 |
| **B1 L2, server → client** | **2,209** | 47,233 | 5,377 | 4,861 | 23,068 | 2.166 | 1,3 |
| B1 L2, src → dst | 2,310 | 47,239 | 5,423 | 4,868 | 23,236 | 2.140 | 1,4 |

### Interpretazione
- **La tratta spiega parte dell'errore:** MAE da 3,30 a 2,21 ms (−33 %).
- **Conta il ruolo, non il verso di registrazione:** server → client è la migliore con entrambi gli identificatori (−0,19 ms sui siti, −0,13 sui prefissi rispetto alla non orientata). Src → dst sta in mezzo: coincide con il ruolo solo per i flow registrati client → server (XRootD, ~82 %).
- **Prefissi L2 migliori dei siti** in tutti gli orientamenti; il vantaggio è maggiore sui flow con sito mancante (vedi sotto).
- **RMSE invariato, RMSE robusto † −7 %:** la tratta non spiega gli estremi.
- **Fallback poco usato:** livello 1 per il 99,7 % (siti) e il 97,8 % (prefissi) dei flow.

### MAE in calo, P90 in aumento: perché
- **P90** = valore sotto cui cade il 90 % degli errori assoluti (ignora il 10 % peggiore); **MAE** = media di tutti gli errori.
- Esempio con 10 flow:

| | Errori | MAE | P90 |
|---|---|---|---|
| B0 (sempre 4,4) | 5 × 0,6 · 5 × 4,4 | 2,5 | 4,4 |
| B1 (mediana per tratta) | 6 × 0,1 · 4 × 5,0 | 2,06 | 5,0 |

- Con la B1 molti flow passano a errore quasi nullo (MAE giù), ma quelli che restano sbagliati sbagliano un po' di più (P90 su). Nei dati: dentro ogni tratta restano flow brevi (~0) e lunghi (~5 ms); quelli dalla parte opposta della mediana della propria tratta sbagliano di ~5 ms, mentre con la mediana globale (4,4) l'errore "tipico" massimo era 4,4. **La tratta da sola non distingue i due regimi.**

### Dettaglio per sottogruppo (server → client)

| Gruppo | MAE B0 | MAE B1 siti | MAE B1 L2 | P90 B0 | P90 B1 L2 |
|---|---|---|---|---|---|
| globale | 3,298 | 2,270 | 2,209 | 4,510 | 4,861 |
| FNAL | 1,636 | 3,419 | 3,396 | 4,381 | 4,806 |
| sito mancante | 3,539 | 2,611 | 2,380 | 4,847 | 4,885 |
| verso incerto | 4,570 | 0,886 | 0,451 | 4,423 | 0,286 |
| flow > 1 min | 3,129 | 2,761 | 2,707 | 4,643 | 5,061 |

Livelli del fallback (B1 L2 server → client): coppia L2 97,8 % (MAE 2,18) · coppia ASN 2,0 % (3,63) · globale 0,16 % (4,54).

- **Sito mancante:** i prefissi recuperano informazione dove il sito manca (2,38 contro 2,61).
- **Verso incerto:** MAE da 4,57 a 0,45; flow su tratte con `delta_rtt` vicino a 0.
- **FNAL peggiora:** vedi §8.

---

## 8. Anomalia FNAL (aperta)

### Fatti
1. **Drift sulle tratte con FNAL come client** (train → validation):

| Tratta (server → FNAL client) | Mediana train | Mediana val | % flow > 1 min train | % flow > 1 min val |
|---|---|---|---|---|
| CERN-PROD | 0,22 | 4,40 | 11 % | 22 % |
| GLOW | 0,72 | 4,82 | 61 % | 76 % |
| MIT_CMS | 0,37 | 5,00 | 37 % | 84 % |
| UCSDT2 | 1,32 | 5,04 | 73 % | 95 % |
| Nebraska | 0,49 | 4,17 | 43 % | 49 % |

2. **Le tratte con FNAL come server sono stabili** (FNAL → CERN-PROD, GRIF, UFlorida-HPC: mediane ~0,01–0,02 ms in entrambi gli split).
3. Il numero di flow nel train **non** è piccolo (es. 188.397 per CERN-PROD → FNAL): il problema è *quali* flow, non *quanti*.

### Ipotesi scartata (per ora)
"Il guasto `end_rtt = 0` colpiva i flow lunghi; la pulizia li ha rimossi e nel train sono rimasti solo flow brevi."
- Nel train, tra i flow registrati con destinazione FNAL, rimossi (`end_rtt = 0`) e rimasti hanno **la stessa quota** di flow oltre 1 minuto (13–17 %), ma durate mediane molto diverse (12–16 s contro 0,2–0,5 s).
- **Limite della verifica:** il drift riguarda FNAL come *client* (colonne canoniche), mentre la verifica sul file grezzo selezionava `dst = FNAL`. Con XRootD (registrato client → server) un flow con FNAL client compare come `src = FNAL`: le popolazioni coincidono solo in parte.

### Andamento nel tempo (FNAL client; ordine cronologico)

| Periodo | Mediana `delta_rtt` | % `end_rtt = 0` (FNAL, qualunque ruolo) |
|---|---|---|
| 8–11 ott (train) | 0,2–0,7 ms | 20–32 % |
| 12 ott, fino alle 17:12 (train) | **3,86 ms** | **0 %** |
| 12 ott, dopo le 17:12 (val) | 0,47 ms | 43 % |
| 13 ott (val) | **4,96 ms** | **0,45 %** |

(Nell'output originale le due righe del 12 ottobre compaiono invertite, per effetto di `ORDER BY split DESC`.)

- La mediana e la quota di `end_rtt = 0` vanno **in direzioni opposte e in modo non graduale**: quando il guasto è attivo la mediana dei flow sopravvissuti è ~0; quando è assente è ~5 ms, come per gli altri siti. Il guasto sembra accendersi e spegnersi.

### Ipotesi da verificare
Quando il guasto è attivo, la RTT finale non viene misurata davvero: a volte arriva come 0 (flow rimossi), a volte forse come un valore "vecchio", vicino a quello iniziale, dando `delta_rtt` ≈ 0 che sembra valido ma non lo è. Se confermata, una parte dei flow FNAL del train ha un **target sbagliato pur non essendo nullo**, e questo riguarda tutti i modelli, non solo la B1.

### Prossime verifiche
1. Andamento **orario** della quota di `end_rtt = 0` e della mediana di `delta_rtt` (FNAL client).
2. Quota di flow con `end_rtt` quasi uguale a `start_rtt` nei periodi con e senza guasto.
3. Decidere come trattare FNAL nei modelli (es. escludere dal train i periodi con guasto attivo, oppure solo documentare e riportare il sottogruppo a parte).
4. Domanda per Carmen: il guasto `end_rtt = 0` verso FNAL è noto? Nei periodi in cui è attivo, anche gli `end_rtt` non nulli sono affidabili?

---

## 9. Prossimi passi

1. Chiudere l'analisi FNAL (§8).
2. `vocab.py`: fit sul train, transform su tutti gli split; ID riservati PAD = 0, MISSING = 1, RARE = 2, UNK = 3; soglia N = 1.000.
3. Aggiungere l'**ora del giorno** come feature (il giorno della settimana no: train giovedì–lunedì, validation lunedì–martedì, test martedì–mercoledì).
4. **B2** ridge su one-hot sparso + `start_rtt`; **B3** MLP senza token (stessa rete che si userà con gli embedding); **B4** LightGBM con categoriche native e loss L1, con feature importance. Addestramento su un campione casuale di ~3 M righe di train, valutazione su tutta la validation.
5. **Diagnostica fuori classifica:** LightGBM con la durata, per misurare l'errore dovuto al non conoscerla. Serve anche per la discussione con Carmen, perché la slide 8 elenca la durata tra gli input mentre il README la vieta.

---

## 10. Punti aperti e domande per Carmen

1. Il picco a ~5 ms di `delta_rtt` nei flow lunghi è un effetto reale (es. coda gestita da fq_codel, obiettivo 5 ms) o una differenza tra le misure di start e di end?
2. Il guasto `end_rtt = 0` verso USCMS-FNAL-WC1 è noto? Quando è attivo, gli `end_rtt` non nulli sono affidabili?
3. Il target è `delta_rtt` (README) o la RTT stessa (slide)?
4. Sito e ASN CRIC sono disponibili al momento della previsione?
5. La durata può essere un input, come nella slide 8?
6. `values.netlink.rtt` è la RTT smussata del kernel Linux?

---

## Appendice – Trappole metodologiche incontrate

- **Leakage temporale:** k-fold con shuffle su dati temporali dà errori ottimisti.
- **Usare il test per decidere:** anche una scelta "innocua" (normalizzazione di `application`) va presa senza il test.
- **Metriche robuste definite sull'errore del modello:** premiano i modelli che sbagliano molto su pochi flow; la definizione va fatta sul valore vero, con soglie del train.
- **Loss diversa dalla metrica:** addestrare con MSE e valutare con MAE significa ottimizzare la cosa sbagliata.
- **Dati mancanti non a caso:** togliere le righe con misura mancante non è neutro se la mancanza dipende dal flow; può introdurre un bias di selezione.
- **Ordine delle righe in SQL:** senza `ORDER BY` non è garantito, anche se "di solito" funziona.
- **Verificare la popolazione giusta:** colonne canoniche (server/client) e colonne grezze (src/dst) non selezionano gli stessi flow.
