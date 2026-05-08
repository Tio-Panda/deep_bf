CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conv2d_init (
    id INTEGER PRIMARY KEY,
    init_weights TEXT NOT NULL,
    init_bias TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activation (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS apod (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS resampler (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS beamformer (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    resampler_id INTEGER NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json)),
    FOREIGN KEY (resampler_id) REFERENCES resampler(id)
);

CREATE TABLE IF NOT EXISTS model (
    id INTEGER PRIMARY KEY,
    family TEXT NOT NULL,
    model_id INTEGER NOT NULL,
    conv2d_init_id INTEGER NOT NULL,
    activation_id INTEGER NOT NULL,
    beamformer_id INTEGER NOT NULL,
    UNIQUE (family, model_id),
    FOREIGN KEY (conv2d_init_id) REFERENCES conv2d_init(id),
    FOREIGN KEY (activation_id) REFERENCES activation(id),
    FOREIGN KEY (beamformer_id) REFERENCES beamformer(id)
);

CREATE TABLE IF NOT EXISTS architecture_cnn_bf (
    model_id INTEGER NOT NULL,
    family TEXT NOT NULL,
    pos INTEGER NOT NULL,
    type TEXT NOT NULL,
    ch_in INTEGER NOT NULL,
    ch_out INTEGER NOT NULL,
    kernel_json TEXT NOT NULL CHECK (json_valid(kernel_json)),
    padding TEXT NOT NULL,
    bias INTEGER NOT NULL CHECK (bias IN (0, 1)),
    PRIMARY KEY (model_id, family, pos)
);

CREATE TABLE IF NOT EXISTS criterion (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS optimizer (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS scheduler (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS hyperparameters (
    id INTEGER PRIMARY KEY,
    seed INTEGER NOT NULL,
    n_epoch INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    learning_rate REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trainloop (
    id INTEGER PRIMARY KEY,
    criterion_id INTEGER NOT NULL,
    optimizer_id INTEGER NOT NULL,
    scheduler_id INTEGER NOT NULL,
    hyperparameters_id INTEGER NOT NULL,
    FOREIGN KEY (criterion_id) REFERENCES criterion(id),
    FOREIGN KEY (optimizer_id) REFERENCES optimizer(id),
    FOREIGN KEY (scheduler_id) REFERENCES scheduler(id),
    FOREIGN KEY (hyperparameters_id) REFERENCES hyperparameters(id)
);

CREATE TABLE IF NOT EXISTS webdataset (
    id INTEGER PRIMARY KEY,
    seed INTEGER NOT NULL,
    mode TEXT NOT NULL,
    rf_transform TEXT NOT NULL,
    nz INTEGER NOT NULL,
    nx INTEGER NOT NULL,
    ns INTEGER NOT NULL,
    ratio REAL NOT NULL,
    "order" TEXT NOT NULL,
    n INTEGER NOT NULL,
    n_train INTEGER NOT NULL,
    n_val INTEGER NOT NULL,
    query_filter TEXT NOT NULL,
    name_filter_json TEXT NOT NULL CHECK (json_valid(name_filter_json)),
    train_idxs TEXT NOT NULL,
    val_idxs TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY,
    version INTEGER NOT NULL,
    webdataset_id INTEGER NOT NULL,
    trainloop_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    commit_hash TEXT NOT NULL DEFAULT 'unknown',
    commit_msg TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (webdataset_id) REFERENCES webdataset(id),
    FOREIGN KEY (trainloop_id) REFERENCES trainloop(id),
    FOREIGN KEY (model_id) REFERENCES model(id)
);

CREATE INDEX IF NOT EXISTS idx_architecture_cnn_bf_family_model_pos
    ON architecture_cnn_bf(family, model_id, pos);

CREATE INDEX IF NOT EXISTS idx_model_beamformer_id
    ON model(beamformer_id);

CREATE INDEX IF NOT EXISTS idx_beamformer_resampler_id
    ON beamformer(resampler_id);

CREATE INDEX IF NOT EXISTS idx_trainloop_criterion_optimizer_scheduler_hyper
    ON trainloop(criterion_id, optimizer_id, scheduler_id, hyperparameters_id);

CREATE INDEX IF NOT EXISTS idx_experiments_refs
    ON experiments(webdataset_id, trainloop_id, model_id);
