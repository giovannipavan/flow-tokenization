# Che progetto è e cosa abbiamo in mano {#intro}

Il progetto predice **come cambia il RTT** di un flusso di rete della WLCG usando solo le informazioni note quando il flusso parte. La tokenizzazione e il pre-training self-supervised sono il contributo originale; la baseline supervisionata è il risultato minimo garantito.

- Il target è `delta_rtt = end_rtt - start_rtt`, con split cronologico train/validation/test.
- La baseline usa feature del start event: IP, porte, protocollo, siti, applicazione e tempo.
- La fase 2 trasforma ogni flusso in una breve sequenza di token discreti.
- La fase 3 pre-addestra un piccolo transformer con masked prediction, poi lo confronta con le baseline.

@answer scope="Dati Firefly/ESnet · 67M flussi · 7 giorni di ottobre 2026 · unità RTT non calibrate"
@meta locale=en quality_profile=showcase

## Quanto è grande il dataset? {#dataset}

| Voce | Valore |
| --- | --- |
| Flussi totali | 67.244.519 |
| Train | 47.071.163 |
| Validation | 10.086.678 |
| Test | 10.086.678 |
| Finestra temporale | 2026-10-07T23:59:59Z → 2026-10-14T23:59:59Z |

Il train finisce il 2026-10-12 alle 17:12 UTC; la validation finisce il 2026-10-13 alle 22:18 UTC; il test inizia dopo. I confini esatti sono in `splits.json`.

## Come è organizzato il lavoro? {#stages}

@workflow id=stages-diagram title="Quattro stadi, dal più semplice al più ambizioso"
| base | Baseline supervisionata | backend |
| token | Tokenizzazione dei flussi | backend |
| pretrain | Pre-training self-supervised | backend |
| compare | Fine-tuning e confronto | backend |
| base > token | |
| token > pretrain | |
| pretrain > compare | |

Ogni stadio è un deliverable completo: se la tokenizzazione si blocca, la baseline resta un progetto valido.

## A cosa serve ciascun file? {#files}

| File | Cosa contiene | Cosa devi farne |
| --- | --- | --- |
| `flows.parquet` | I 67M flussi, circa 4–5 GB | È il dato vero; resta fuori da git |
| `manifest.json` | Schema, versione, validazione e SHA-256 del parquet | Verifica integrità e tipi |
| `splits.json` | Confini cronologici di train/validation/test | Usalo per dividere i dati |
| `README` del dataset | Descrizione completa dei campi | È già riassunto in `AGENTS.md` |

## Quali colonne non posso usare? {#leakage}

@callout id=leak level=warning title="Regola di leakage"
Non usare mai `end_rtt`, `delta_rtt`, `flow_end`, la durata derivata, `endpoints_match`, `endpoints_reversed` e `start_fields_match`. Sono noti solo dopo la fine del flusso e falserebbero la valutazione.

`start_rtt` è disponibile solo dopo la prima osservazione RTT: decidi esplicitamente se usarlo e documenta la scelta. Tokenizer, scaler, imputer e selezione feature si addestrano **solo sul train**.

## Cosa fare adesso con i file? {#checklist}

- [ ] Copia `flows.parquet`, `manifest.json`, `splits.json` e il README in `data/`, oppure in una cartella esterna e imposta `FLOWS_DATA_DIR`.
- [ ] Calcola lo SHA-256 di `flows.parquet` e confrontalo con quello dentro `manifest.json`.
- [ ] Apri `manifest.json` e controlla schema, tipi e risultati di validazione.
- [ ] Leggi `splits.json` e conferma i tre confini temporali.
- [ ] Esegui una query DuckDB di prova con `SET TimeZone = 'UTC'`.
- [ ] Controlla che `data/` e `temp-diagrams/` siano ignorati da git prima di committare.
