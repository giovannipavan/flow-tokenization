"""flowtok: network flow tokenization project (CS-433, EPFL).

Modules:
    paths          project folders and file locations
    db             DuckDB connection and views on the data files
    prepare        encoding of flows_prepared.parquet into data/encoded/
    data           loading of the encoded sets as arrays
    evaluation     metrics shared by all models
    models         one module per baseline (b0, b1, b2, b4)
    run_baselines  train, evaluate and save the models listed in a config
    tune           hyperparameter search on the tune set
"""
