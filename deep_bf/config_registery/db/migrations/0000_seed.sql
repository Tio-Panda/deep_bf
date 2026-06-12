BEGIN;

CREATE TABLE IF NOT EXISTS conv2d_init_config (
    id INTEGER PRIMARY KEY,
    init_weights TEXT NOT NULL,
    init_bias TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activation_config (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS architecture_cnn_bf_config (
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

CREATE TABLE IF NOT EXISTS criterion_config (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS optimizer_config (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS scheduler_config (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS hyperparameters_config (
    id INTEGER PRIMARY KEY,
    seed INTEGER NOT NULL,
    n_epoch INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    learning_rate REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS data_type_config (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS beamformer_config (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS resampler_config (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS compounding_config (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS apod_config (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS data_size_config (
    id INTEGER PRIMARY KEY,
    nz INTEGER NOT NULL,
    nx INTEGER NOT NULL,
    ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS data_preprocessing_config (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE IF NOT EXISTS samples_organization_config (
    id INTEGER PRIMARY KEY,
    seed INTEGER NOT NULL,
    ratio REAL NOT NULL,
    "order" TEXT NOT NULL,
    select_mode TEXT NOT NULL,
    n_train INTEGER NOT NULL,
    n_val INTEGER NOT NULL,
    query TEXT NOT NULL,
    train_idxs TEXT NOT NULL,
    val_idxs TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS beamformer_setup (
    id INTEGER PRIMARY KEY,
    data_type_config_id INTEGER NOT NULL,
    beamformer_config_id INTEGER NOT NULL,
    resampler_config_id INTEGER NOT NULL,
    compounding_config_id INTEGER NOT NULL,
    apod_config_id INTEGER NOT NULL,
    FOREIGN KEY (data_type_config_id) REFERENCES data_type_config(id),
    FOREIGN KEY (beamformer_config_id) REFERENCES beamformer_config(id),
    FOREIGN KEY (resampler_config_id) REFERENCES resampler_config(id),
    FOREIGN KEY (compounding_config_id) REFERENCES compounding_config(id),
    FOREIGN KEY (apod_config_id) REFERENCES apod_config(id)
);

CREATE TABLE IF NOT EXISTS trainloop_setup (
    id INTEGER PRIMARY KEY,
    criterion_config_id INTEGER NOT NULL,
    optimizer_config_id INTEGER NOT NULL,
    scheduler_config_id INTEGER NOT NULL,
    hyperparameters_config_id INTEGER NOT NULL,
    FOREIGN KEY (criterion_config_id) REFERENCES criterion_config(id),
    FOREIGN KEY (optimizer_config_id) REFERENCES optimizer_config(id),
    FOREIGN KEY (scheduler_config_id) REFERENCES scheduler_config(id),
    FOREIGN KEY (hyperparameters_config_id) REFERENCES hyperparameters_config(id)
);

CREATE TABLE IF NOT EXISTS model_pack (
    id INTEGER PRIMARY KEY,
    family TEXT NOT NULL,
    model_id INTEGER NOT NULL,
    conv2d_init_config_id INTEGER NOT NULL,
    activation_config_id INTEGER NOT NULL,
    beamformer_setup_id INTEGER NOT NULL,
    UNIQUE (family, model_id),
    FOREIGN KEY (conv2d_init_config_id) REFERENCES conv2d_init_config(id),
    FOREIGN KEY (activation_config_id) REFERENCES activation_config(id),
    FOREIGN KEY (beamformer_setup_id) REFERENCES beamformer_setup(id)
);

CREATE TABLE IF NOT EXISTS webdataset_beamformer_pack (
    id INTEGER PRIMARY KEY,
    beamformer_setup_id INTEGER NOT NULL,
    data_size_config_id INTEGER NOT NULL,
    data_preprocessing_config_id INTEGER NOT NULL,
    samples_organization_config_id INTEGER NOT NULL,
    FOREIGN KEY (beamformer_setup_id) REFERENCES beamformer_setup(id),
    FOREIGN KEY (data_size_config_id) REFERENCES data_size_config(id),
    FOREIGN KEY (data_preprocessing_config_id) REFERENCES data_preprocessing_config(id),
    FOREIGN KEY (samples_organization_config_id) REFERENCES samples_organization_config(id)
);

CREATE TABLE IF NOT EXISTS experiment (
    id INTEGER PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    model_pack_id INTEGER NOT NULL,
    trainloop_setup_id INTEGER NOT NULL,
    webdataset_beamformer_pack_id INTEGER NOT NULL,
    FOREIGN KEY (model_pack_id) REFERENCES model_pack(id),
    FOREIGN KEY (trainloop_setup_id) REFERENCES trainloop_setup(id),
    FOREIGN KEY (webdataset_beamformer_pack_id) REFERENCES webdataset_beamformer_pack(id)
);

CREATE INDEX IF NOT EXISTS idx_architecture_cnn_bf_config_family_model_pos
    ON architecture_cnn_bf_config(family, model_id, pos);

CREATE INDEX IF NOT EXISTS idx_model_pack_beamformer_setup_id
    ON model_pack(beamformer_setup_id);

CREATE INDEX IF NOT EXISTS idx_webdataset_beamformer_pack_refs
    ON webdataset_beamformer_pack(
        beamformer_setup_id,
        data_size_config_id,
        data_preprocessing_config_id,
        samples_organization_config_id
    );

CREATE INDEX IF NOT EXISTS idx_experiment_refs
    ON experiment(model_pack_id, trainloop_setup_id, webdataset_beamformer_pack_id);

INSERT INTO conv2d_init_config (id, init_weights, init_bias) VALUES (0, 'XavierUniform', 'Zeros');

INSERT INTO activation_config (id, type, params_json) VALUES (0, 'LeakyReLU', '{"negative_slope": 0.01}');

INSERT INTO resampler_config (id, type, params_json) VALUES (0, 'LinearInterpolation', '{}');
INSERT INTO resampler_config (id, type, params_json) VALUES (1, 'GridSample', '{"align_corners": false, "mode": "bilinear", "padding_mode": "zeros"}');
INSERT INTO resampler_config (id, type, params_json) VALUES (2, 'GridSample', '{"align_corners": false, "mode": "bilinear", "padding_mode": "border"}');
INSERT INTO resampler_config (id, type, params_json) VALUES (3, 'GridSample', '{"align_corners": true, "mode": "bilinear", "padding_mode": "zeros"}');
INSERT INTO resampler_config (id, type, params_json) VALUES (4, 'GridSample', '{"align_corners": true, "mode": "bilinear", "padding_mode": "border"}');

INSERT INTO apod_config (id, type, params_json) VALUES (0, 'None', '{}');
INSERT INTO apod_config (id, type, params_json) VALUES (1, 'DynamicReceiveAperture', '{"f_num": 1, "window": "boxcar"}');
INSERT INTO apod_config (id, type, params_json) VALUES (2, 'DynamicReceiveAperture', '{"f_num": 1.75, "window": "hanning"}');
INSERT INTO apod_config (id, type, params_json) VALUES (3, 'DynamicReceiveAperture', '{"f_num": 1.75, "window": "hamming"}');
INSERT INTO apod_config (id, type, params_json) VALUES (4, 'DynamicReceiveAperture', '{"f_num": 1.75, "window": "tukey25"}');

INSERT INTO beamformer_config ("id", "type", "params_json") VALUES (0, 'DAS', '{"batch_size": 128}');
INSERT INTO beamformer_config ("id", "type", "params_json") VALUES (1, 'MV', '{ "batch_size": 1, "z_chunk": 1024, "L": 32, "temporal_radius": 3, "eps": 1e-10 }');
INSERT INTO beamformer_config ("id", "type", "params_json") VALUES (2, 'F-DMAS', '{ "BW": 0.5, "transition_low_ratio": 0.6, "transition_high_ratio": 0.4, "ripple": 1e-3, "batch_size": 128, "eps": 1e-10, "min_band_bins": 4 }');
INSERT INTO beamformer_config ("id", "type", "params_json") VALUES (3, 'CF', '{"batch_size": 128, "eps": 1e-08}');
INSERT INTO beamformer_config ("id", "type", "params_json") VALUES (4, 'iMAP', '{"batch_size":128, "num_iters":2, "eps":1e-8}');
INSERT INTO beamformer_config ("id", "type", "params_json") VALUES (5, 'SparseRegularization', '{"batch_size": 128, "eps": 1e-08, "lam": 0.001, "num_iters": 20, "step": 1.0}');
INSERT INTO beamformer_config ("id", "type", "params_json") VALUES (6, 'SparseRegularization2', '{"J": 2, "batch_size": 128, "eps": 1e-08, "lam": 0.0001, "mode": "symmetric", "num_iters": 15, "sigma": 0.2, "tau": 0.2, "theta": 1.0}');

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 0, 'BF', 1, 1, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 1, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 2, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 5, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 1, 'BF', 16, 16, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 2, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 5, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 2, 'BF', 8, 8, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 5, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 3, 'BF', 8, 8, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 5, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 3, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 4, 'BF', 4, 4, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 5, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 3, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 4, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 5, 'BF', 2, 2, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 3, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 4, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 5, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 6, 'BF', 1, 1, '[-1, -1]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 0, 'BF', 1, 1, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 1, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 2, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 5, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 1, 'BF', 16, 16, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 2, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 5, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 2, 'BF', 8, 8, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 5, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 3, 'BF', 8, 8, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 5, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 3, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 4, 'BF', 4, 4, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 5, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 3, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 4, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);
INSERT INTO architecture_cnn_bf_config (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 5, 'BF', 1, 1, '[-1, -1]', 'same', 1);

INSERT INTO criterion_config (id, type, params_json) VALUES (0, 'MSE', '{"reduction": "mean"}');
INSERT INTO criterion_config (id, type, params_json) VALUES (1, 'MSLE', '{"reduction": "mean"}');

INSERT INTO optimizer_config (id, type, params_json) VALUES (0, 'Adam', '{"amsgrad": false, "betas": [0.9, 0.999], "eps": 1e-07, "weight_decay": 0.0}');

INSERT INTO scheduler_config (id, type, params_json) VALUES (0, 'ReduceLROnPlateau', '{"factor": 0.5, "min_lr": 1e-06, "mode": "min", "patience": 5, "threshold": 0.0001}');

INSERT INTO hyperparameters_config (id, seed, n_epoch, batch_size, learning_rate) VALUES (0, 42, 100, 4, 1e-3);
INSERT INTO hyperparameters_config (id, seed, n_epoch, batch_size, learning_rate) VALUES (1, 42, 100, 2, 1e-3);

INSERT INTO data_preprocessing_config (id, type, params_json) VALUES (0, 'none', '{}');
INSERT INTO data_preprocessing_config (id, type, params_json) VALUES (1, 'sharifzadeh', '{"eps": 1e-8}');

INSERT INTO samples_organization_config (id, seed, ratio, "order", select_mode, n_train, n_val, query, train_idxs, val_idxs) VALUES
    (0, 42, 0.9, 'CWH', 'select_idxs', 495, 55, '(RF == 1) and (nc == 128) and (name.str.slice(0, 3) != ''JHU'')', '0:12,14:19,22:51,53:57,59:70,72:86,89:90,92:98,100:101,103:105,107:120,122:129,131:159,161:165,167:188,190,192:213,215:240,242,244:251,253:263,265:269,271:272,274:275,277:307,309:312,314,316:329,331:338,340:342,344,346:365,367:371,373:384,386,388:412,414:426,428:434,436:442,444:453,455:457,460:465,467:473,476:483,485:490,492:503,506:509,511:529,531:534,536:543,545:549', '13,20:21,52,58,71,87:88,91,99,102,106,121,130,160,166,189,191,214,241,243,252,264,270,273,276,308,313,315,330,339,343,345,366,372,385,387,413,427,435,443,454,458:459,466,474:475,484,491,504:505,510,530,535,544'),
    (1, 42, 0.9, 'CWH', 'random_split', -1, -1, '(RF == 1) and (nc == 128) and (name.str.slice(0, 3) != ''JHU'') and (source == ''CUBDL'')', '-1', '-1');

INSERT INTO data_size_config (id, nz, nx, ns) VALUES
    (0, 2048, 256, 2300),
    (1, 2048, 256, 2800),
    (2, 1024, 256, 2300),
    (3, 512, 256, 2300),
    (4, 256, 256, 2300),
    (5, -1, -1, 2300);


INSERT INTO data_type_config (id, type, params_json) VALUES
    (0, 'RF', '{}'),
    (1, 'RF Analitic', '{}'),
    (2, 'IQ', '{}');

INSERT INTO compounding_config (id, type, params_json) VALUES (0, 'NONE', '{}');
INSERT INTO compounding_config ("id", "type", "params_json") VALUES (1, 'CoherentPlane-WaveCompoundingWithMean', '{}');
INSERT INTO compounding_config ("id", "type", "params_json") VALUES (2, 'CoherentPlane-WaveCompoundingAngularApodizationShortPulse', '{"kind": "hanning"}');
INSERT INTO compounding_config ("id", "type", "params_json") VALUES (3, 'CoherentPlane-WaveCompoundingAngularApodizationParaxial', '{"kind": "hanning"}');
INSERT INTO compounding_config ("id", "type", "params_json") VALUES (4, 'CoherentPlane-WaveCompoundingWithEDT', '{"alpha": 0.001, "eps": 1e-08}');


INSERT INTO beamformer_setup (id, data_type_config_id, beamformer_config_id, resampler_config_id, compounding_config_id, apod_config_id) VALUES 
    (0, 0, 0, 1, 1, 0),
    (1, 0, 1, 1, 1, 0),
    (2, 0, 2, 1, 1, 0),
    (3, 0, 3, 1, 1, 0),
    (4, 0, 4, 1, 1, 0),
    (5, 0, 5, 1, 1, 0),
    (6, 0, 0, 1, 2, 0),
    (7, 0, 1, 1, 2, 0),
    (8, 0, 2, 1, 2, 0),
    (9, 0, 3, 1, 2, 0),
    (10, 0, 4, 1, 2, 0),
    (11, 0, 5, 1, 2, 0),
    (12, 0, 0, 1, 4, 0),
    (13, 0, 1, 1, 4, 0),
    (14, 0, 2, 1, 4, 0),
    (15, 0, 3, 1, 4, 0),
    (16, 0, 4, 1, 4, 0),
    (17, 0, 5, 1, 4, 0);

INSERT INTO trainloop_setup (id, criterion_config_id, optimizer_config_id, scheduler_config_id, hyperparameters_config_id)
VALUES (0, 0, 0, 0, 0);

INSERT INTO model_pack (id, family, model_id, conv2d_init_config_id, activation_config_id, beamformer_setup_id) VALUES
    (0, 'BINN_OG', 1, 0, 0, 0),
    (1, 'BINN_OG', 2, 0, 0, 0),
    (2, 'BINN_OG', 3, 0, 0, 0),
    (3, 'BINN_OG', 4, 0, 0, 0),
    (4, 'BINN_OG', 5, 0, 0, 0),
    (5, 'BINN_OG', 6, 0, 0, 0),
    (6, 'BINN', 1, 0, 0, 0),
    (7, 'BINN', 2, 0, 0, 0),
    (8, 'BINN', 3, 0, 0, 0),
    (9, 'BINN', 4, 0, 0, 0),
    (10, 'BINN', 5, 0, 0, 0),
    (11, 'BINN', 6, 0, 0, 0),
    (12, 'BINN', 7, 0, 0, 0);

INSERT INTO webdataset_beamformer_pack (id, beamformer_setup_id, data_size_config_id, data_preprocessing_config_id, samples_organization_config_id) VALUES
    (0, 0, 0, 1, 0),
    (1, 1, 0, 1, 0),
    (2, 2, 0, 1, 0),
    (3, 3, 0, 1, 0),
    (4, 4, 0, 1, 0),
    (5, 5, 0, 1, 0),
    (6, 6, 0, 1, 0),
    (7, 7, 0, 1, 0),
    (8, 8, 0, 1, 0),
    (9, 9, 0, 1, 0),
    (10, 10, 0, 1, 0),
    (11, 11, 0, 1, 0),
    (12, 12, 0, 1, 0),
    (13, 13, 0, 1, 0),
    (14, 14, 0, 1, 0),
    (15, 15, 0, 1, 0),
    (16, 16, 0, 1, 0),
    (17, 17, 0, 1, 0);

INSERT INTO experiment (
    id,
    model_pack_id,
    trainloop_setup_id,
    webdataset_beamformer_pack_id,
    description
) VALUES
    (0, 2, 0, 0, 'BINN_OG-3 DAS|CPWC'),
    (1, 2, 0, 1, 'BINN_OG-3 MV|CPWC'),
    (2, 2, 0, 2, 'BINN_OG-3 FDMAS|CPWC'),
    (3, 2, 0, 3, 'BINN_OG-3 CF|CPWC'),
    (4, 2, 0, 4, 'BINN_OG-3 IMAP|CPWC'),
    (5, 2, 0, 5, 'BINN_OG-3 SR|CPWC'),
    (6, 2, 0, 6, 'BINN_OG-3 DAS|AASP'),
    (7, 2, 0, 7, 'BINN_OG-3 MV|AASP'),
    (8, 2, 0, 8, 'BINN_OG-3 FDMAS|AASP'),
    (9, 2, 0, 9, 'BINN_OG-3 CF|AASP'),
    (10, 2, 0, 10, 'BINN_OG-3 IMAP|AASP'),
    (11, 2, 0, 11, 'BINN_OG-3 SR|AASP'),
    (12, 2, 0, 12, 'BINN_OG-3 DAS|EDT'),
    (13, 2, 0, 13, 'BINN_OG-3 MV|EDT'),
    (14, 2, 0, 14, 'BINN_OG-3 FDMAS|EDT'),
    (15, 2, 0, 15, 'BINN_OG-3 CF|EDT'),
    (16, 2, 0, 16, 'BINN_OG-3 IMAP|EDT'),
    (17, 2, 0, 17, 'BINN_OG-3 SR|EDT');

COMMIT;
