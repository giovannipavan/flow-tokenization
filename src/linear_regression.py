"""
prepara_encoded.py — Codifica dei flow per i modelli B2, B3, B4.

Parte da flows_prepared.parquet e produce, in data/encoded/:
    train_enc.parquet   ~3 M flow dal primo 90 % del train   (addestramento)
    tune_enc.parquet    ~1 M flow dall'ultimo 10 % del train (early stopping, scelte)
    val_enc.parquet     tutta la validation, 10,07 M flow    (risultati)
    mappa.parquet       tabella vocabolario / valore / codice
    vocabolari.json     vocabolari valore -> codice, codici riservati, N_MIN
    meta.json           parametri usati, confine del tune, conteggi, dimensioni dei vocabolari
    qualita.csv         quote MISSING / RARE / UNK per colonna e per insieme

È la §5 del notebook delle baseline (5.1–5.5), con quattro correzioni:
    - controllo che nelle colonne usate non ci siano NULL (che finirebbero in UNK invece che in MISSING);
    - codici a pari frequenza assegnati in ordine alfabetico, così i vocabolari sono sempre uguali;
    - `riga` numerata in ordine di (t_start, flow_id), quindi stabile tra esecuzioni;
    - flow_id e t_start tenuti nei file, come chiave per riattaccare le previsioni ai flow.

Il test non viene mai letto.
Uso:  python prepara_encoded.py   (da qualunque cartella del progetto)
"""

import json
import time
from pathlib import Path

import duckdb
import pandas as pd


# =============================================================================
# Parametri
# =============================================================================

N_MIN   = 1000          # flow minimi nel train perché un valore abbia un codice suo
N_TRAIN = 3_000_000     # dimensione obiettivo di train_enc
N_TUNE  = 1_000_000     # dimensione obiettivo di tune_enc
QUOTA_TUNE = 0.9        # il tune inizia dopo il 90 % dei flow del train (in ordine di tempo)

PAD, MISSING, RARE, UNK = 0, 1, 2, 3

# Campionamento riproducibile: flow tenuto se il suo hash cade sotto una soglia.
# hash() è quello usato nel notebook; il campione è lo stesso con la stessa versione di DuckDB.
# Una volta salvati i file, tutto il gruppo usa quelli: la versione non conta più.
HASH = "hash(f.flow_id || CAST(f.t_start AS VARCHAR)) % 100000"

# vocabolario -> colonne che lo usano (siti e reti L2: vocabolario condiviso server/client)
VOCAB_COLONNE = {
    "ip_version":     ["ip_version"],
    "experiment":     ["experiment"],
    "exp_act":        ["exp_act"],
    "application":    ["application"],
    "hostname":       ["hostname"],
    "site":           ["server_site", "client_site"],
    "L2":             ["server_L2", "client_L2"],
    "porta_servizio": ["porta_servizio"],
    "porta_client":   ["porta_client"],
    "verso":          ["verso"],
    "verso_incerto":  ["verso_incerto"],
}
FEATURE_CAT = [c for colonne in VOCAB_COLONNE.values() for c in colonne]
FEATURE_NUM = ["start_rtt"]
TARGET      = "delta_rtt"

# colonne in chiaro tenute SOLO per la valutazione per sottogruppi: NON sono input dei modelli
COLONNE_VALUTAZIONE = ["server_site", "client_site", "verso_incerto", "an_durata_s"]


# =============================================================================
# Percorsi e connessione
# =============================================================================

def trova_root():
    """Risale dalle cartelle dello script finché trova data/student_dataset."""
    qui = Path(__file__).resolve()
    for p in [qui.parent, *qui.parents]:
        if (p / "data" / "student_dataset").is_dir():
            return p
    raise FileNotFoundError("Cartella data/student_dataset non trovata sopra lo script.")


PROJECT_ROOT = trova_root()
DATA_DIR = PROJECT_ROOT / "data" / "student_dataset"
PREPARED = DATA_DIR / "flows_prepared.parquet"
ENC_DIR  = PROJECT_ROOT / "data" / "encoded"


def connetti():
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit = '8GB'")
    con.execute(f"SET temp_directory = '{(PROJECT_ROOT / 'duckdb_tmp').as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"CREATE OR REPLACE VIEW fp AS SELECT * FROM read_parquet('{PREPARED.as_posix()}')")
    return con


# =============================================================================
# Passi
# =============================================================================

def controlla_null(con):
    """Nessun NULL nelle colonne usate (un NULL non troverebbe mai la sua riga nella mappa)."""
    colonne = FEATURE_CAT + FEATURE_NUM + [TARGET, "flow_id", "t_start"]
    conteggi = ", ".join(f"COUNT(*) FILTER (WHERE {c} IS NULL) AS {c}" for c in colonne)
    nulli = con.execute(f"SELECT {conteggi} FROM fp WHERE split IN ('train', 'val')").df().iloc[0]
    nulli = nulli[nulli > 0]
    if len(nulli) > 0:
        raise ValueError(f"Valori NULL trovati (sostituirli con 'MISSING' a monte):\n{nulli}")
    print("Controllo NULL: nessun NULL nelle colonne usate.")


def costruisci_vocabolari(con):
    """Per ogni vocabolario: valori con >= N_MIN flow nel train -> codici da 4, per frequenza."""
    vocab, visti = {}, {}
    for nome, colonne in VOCAB_COLONNE.items():
        parti = [f"SELECT CAST({c} AS VARCHAR) AS valore FROM fp WHERE split = 'train'"
                 for c in colonne]
        conteggi = con.execute(f"""
            SELECT valore, COUNT(*) AS n
            FROM ({" UNION ALL ".join(parti)})
            GROUP BY valore
            ORDER BY n DESC, valore          -- a pari frequenza, ordine alfabetico: deterministico
        """).df()

        frequenti = conteggi.loc[(conteggi["n"] >= N_MIN) & (conteggi["valore"] != "MISSING"), "valore"]
        vocab[nome] = {v: i + 4 for i, v in enumerate(frequenti)}
        visti[nome] = set(conteggi["valore"])
        print(f"  {nome:15s} valori nel train: {len(conteggi):6d} | con >= {N_MIN} flow: {len(vocab[nome]):5d}")
    return vocab, visti


def crea_mappa(con, vocab, visti):
    """Tabella DuckDB (nome, valore, codice) per tutti i valori visti nel train."""
    righe = []
    for nome in VOCAB_COLONNE:
        for valore in sorted(visti[nome]):
            if valore == "MISSING":
                codice = MISSING
            elif valore in vocab[nome]:
                codice = vocab[nome][valore]
            else:
                codice = RARE
            righe.append((nome, valore, codice))
    mappa_df = pd.DataFrame(righe, columns=["nome", "valore", "codice"])
    con.register("mappa_df", mappa_df)
    con.execute("CREATE OR REPLACE TABLE mappa AS SELECT * FROM mappa_df")
    print(f"  righe nella mappa: {len(mappa_df):,}")
    return mappa_df


def query_codificata(condizione):
    """Query che restituisce i flow che soddisfano `condizione`, con un codice intero per categorica."""
    colonne_id, join = [], []
    coppie = [(n, c) for n, cols in VOCAB_COLONNE.items() for c in cols]
    for k, (nome, c) in enumerate(coppie):
        colonne_id.append(f"CAST(COALESCE(m{k}.codice, {UNK}) AS INTEGER) AS id_{c}")
        join.append(f"LEFT JOIN mappa AS m{k} "
                    f"ON m{k}.nome = '{nome}' AND m{k}.valore = CAST(f.{c} AS VARCHAR)")
    chiaro = ", ".join(f"f.{c}" for c in COLONNE_VALUTAZIONE)
    return f"""
        SELECT ROW_NUMBER() OVER (ORDER BY f.t_start, f.flow_id) AS riga,
               f.flow_id, f.t_start,
               f.{TARGET}, f.start_rtt,
               {", ".join(colonne_id)},
               {chiaro}
        FROM fp AS f
        {" ".join(join)}
        WHERE {condizione}
    """


def condizioni_insiemi(con):
    """Confine del tune e condizioni SQL dei tre insiemi."""
    tune_start = con.execute(f"""
        SELECT quantile_disc(epoch(t_start), {QUOTA_TUNE}) FROM fp WHERE split = 'train'
    """).fetchone()[0]

    n_pre, n_tune = con.execute(f"""
        SELECT COUNT(*) FILTER (WHERE epoch(t_start) <  {tune_start}),
               COUNT(*) FILTER (WHERE epoch(t_start) >= {tune_start})
        FROM fp WHERE split = 'train'
    """).fetchone()

    soglia_train = round(N_TRAIN / n_pre  * 100_000)
    soglia_tune  = round(N_TUNE  / n_tune * 100_000)

    condizioni = {
        "train_enc": f"f.split = 'train' AND epoch(f.t_start) <  {tune_start} AND {HASH} < {soglia_train}",
        "tune_enc":  f"f.split = 'train' AND epoch(f.t_start) >= {tune_start} AND {HASH} < {soglia_tune}",
        "val_enc":   "f.split = 'val'",
    }
    info = {
        "tune_start_epoch": float(tune_start),
        "tune_start_utc":   str(pd.to_datetime(tune_start, unit="s", utc=True)),
        "n_train_primo_90": int(n_pre),
        "n_train_ultimo_10": int(n_tune),
        "soglia_hash_train": soglia_train,
        "soglia_hash_tune":  soglia_tune,
    }
    print(f"  inizio del tune: {info['tune_start_utc']} | soglie hash: {soglia_train} / {soglia_tune} su 100.000")
    return condizioni, info


def scrivi_insiemi(con, condizioni):
    """Scrive i tre insiemi direttamente da DuckDB in parquet (senza passare da Python)."""
    righe = {}
    for nome, cond in condizioni.items():
        out = (ENC_DIR / f"{nome}.parquet").as_posix()
        t0 = time.time()
        con.execute(f"COPY ({query_codificata(cond)}) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
        righe[nome] = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
        print(f"  {nome:10s} {righe[nome]:>12,} righe  ({time.time() - t0:.1f} s)  -> {out}")
    return righe


def controlli_qualita(con, vocab):
    """Quote MISSING / RARE / UNK per colonna e insieme; codici frequenti senza esempi in train_enc."""
    tabelle = []
    for nome in ["train_enc", "tune_enc", "val_enc"]:
        f = (ENC_DIR / f"{nome}.parquet").as_posix()
        parti = [f"""
            SELECT '{nome}' AS insieme, '{c}' AS colonna,
                   100 * AVG(CASE WHEN id_{c} = {MISSING} THEN 1 ELSE 0 END) AS pct_missing,
                   100 * AVG(CASE WHEN id_{c} = {RARE}    THEN 1 ELSE 0 END) AS pct_rare,
                   100 * AVG(CASE WHEN id_{c} = {UNK}     THEN 1 ELSE 0 END) AS pct_unk
            FROM read_parquet('{f}')""" for c in FEATURE_CAT]
        tabelle.append(con.execute(" UNION ALL ".join(parti)).df())
    qualita = pd.concat(tabelle, ignore_index=True)

    # Codici "frequenti" (>= N_MIN nel train completo) che in train_enc non compaiono mai:
    # il modello non ha esempi per impararne il coefficiente/embedding.
    f_train = (ENC_DIR / "train_enc.parquet").as_posix()
    senza_esempi = {}
    for nome, colonne in VOCAB_COLONNE.items():
        parti = [f"SELECT id_{c} AS id FROM read_parquet('{f_train}')" for c in colonne]
        presenti = con.execute(f"""
            SELECT COUNT(DISTINCT id) FROM ({" UNION ALL ".join(parti)}) WHERE id >= 4
        """).fetchone()[0]
        senza_esempi[nome] = len(vocab[nome]) - int(presenti)
    return qualita, senza_esempi


# =============================================================================
# Main
# =============================================================================

def main():
    t_inizio = time.time()
    print(f"Progetto: {PROJECT_ROOT}")
    if not PREPARED.exists():
        raise FileNotFoundError(PREPARED)
    ENC_DIR.mkdir(parents=True, exist_ok=True)

    con = connetti()
    print("DuckDB", duckdb.__version__)

    print("\n[1/6] Controllo NULL")
    controlla_null(con)

    print("\n[2/6] Vocabolari (sul train completo)")
    vocab, visti = costruisci_vocabolari(con)

    print("\n[3/6] Tabella mappa")
    mappa_df = crea_mappa(con, vocab, visti)
    mappa_df.to_parquet(ENC_DIR / "mappa.parquet", index=False)

    print("\n[4/6] Confine del tune e campioni")
    condizioni, info_tune = condizioni_insiemi(con)

    print("\n[5/6] Scrittura degli insiemi codificati")
    righe = scrivi_insiemi(con, condizioni)

    print("\n[6/6] Controlli di qualità")
    qualita, senza_esempi = controlli_qualita(con, vocab)
    qualita.round(4).to_csv(ENC_DIR / "qualita.csv", index=False)
    with pd.option_context("display.max_rows", 100, "display.width", 120):
        print(qualita.round(3).to_string(index=False))
    print("Codici frequenti senza esempi in train_enc:", senza_esempi)

    # --- Vocabolari e metadati in JSON ---
    with open(ENC_DIR / "vocabolari.json", "w", encoding="utf-8") as f:
        json.dump({
            "riservati": {"PAD": PAD, "MISSING": MISSING, "RARE": RARE, "UNK": UNK},
            "N_MIN": N_MIN,
            "colonne": VOCAB_COLONNE,
            "vocabolari": vocab,
        }, f, ensure_ascii=False, indent=2)

    meta = {
        "sorgente": PREPARED.name,
        "duckdb": duckdb.__version__,
        "target": TARGET,
        "feature_num": FEATURE_NUM,
        "feature_cat": FEATURE_CAT,
        "colonne_id": ["id_" + c for c in FEATURE_CAT],
        "colonne_solo_valutazione": COLONNE_VALUTAZIONE,
        # dimensione delle tabelle di embedding: 4 codici riservati + valori frequenti
        "n_codici": {nome: 4 + len(v) for nome, v in vocab.items()},
        "N_MIN": N_MIN, "N_TRAIN": N_TRAIN, "N_TUNE": N_TUNE, "QUOTA_TUNE": QUOTA_TUNE,
        "hash": HASH,
        **info_tune,
        "righe": {k: int(v) for k, v in righe.items()},
        "codici_frequenti_senza_esempi_in_train_enc": senza_esempi,
    }
    with open(ENC_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nFatto in {time.time() - t_inizio:.0f} s. File in {ENC_DIR}")


if __name__ == "__main__":
    main()