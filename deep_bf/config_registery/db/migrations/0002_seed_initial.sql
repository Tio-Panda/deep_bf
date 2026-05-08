-- Seed generated from deep_bf/wrapper/config CSV files and cambio.md.
-- Includes criterion MSLE previously added by migration 0003.
BEGIN;

INSERT INTO conv2d_init (id, init_weights, init_bias) VALUES (0, 'XavierUniform', 'Zeros');

INSERT INTO activation (id, type, params_json) VALUES (0, 'LeakyReLU', '{"negative_slope": 0.01}');

INSERT INTO resampler (id, type, params_json) VALUES (0, 'LinearInterpolation', '{}');
INSERT INTO resampler (id, type, params_json) VALUES (1, 'GridSample', '{"align_corners": false, "mode": "bilinear", "padding_mode": "zeros"}');
INSERT INTO resampler (id, type, params_json) VALUES (2, 'GridSample', '{"align_corners": false, "mode": "bilinear", "padding_mode": "border"}');
INSERT INTO resampler (id, type, params_json) VALUES (3, 'GridSample', '{"align_corners": true, "mode": "bilinear", "padding_mode": "zeros"}');
INSERT INTO resampler (id, type, params_json) VALUES (4, 'GridSample', '{"align_corners": true, "mode": "bilinear", "padding_mode": "border"}');

INSERT INTO apod (id, type, params_json) VALUES (0, 'None', '{}');
INSERT INTO apod (id, type, params_json) VALUES (1, 'DynamicReceiveAperture', '{"f_num": 1, "window": "boxcar"}');
INSERT INTO apod (id, type, params_json) VALUES (2, 'DynamicReceiveAperture', '{"f_num": 1.75, "window": "hanning"}');
INSERT INTO apod (id, type, params_json) VALUES (3, 'DynamicReceiveAperture', '{"f_num": 1.75, "window": "hamming"}');
INSERT INTO apod (id, type, params_json) VALUES (4, 'DynamicReceiveAperture', '{"f_num": 1.75, "window": "tukey25"}');

INSERT INTO beamformer (id, type, resampler_id, params_json) VALUES (0, 'DAS', 1, '{}');
INSERT INTO beamformer (id, type, resampler_id, params_json) VALUES (1, 'MVB', 1, '{"z_chunk": 1024, "L": 16, "diagonal_loading": 1e-3, "eps": 1e-10}');

INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (0, 'BINN', 1, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (1, 'BINN', 2, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (2, 'BINN', 3, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (3, 'BINN', 4, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (4, 'BINN', 5, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (5, 'BINN', 6, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (6, 'BINN', 7, 0, 0, 0);

INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (7, 'BINN_OG', 1, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (8, 'BINN_OG', 2, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (9, 'BINN_OG', 3, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (10, 'BINN_OG', 4, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (11, 'BINN_OG', 5, 0, 0, 0);
INSERT INTO model (id, family, model_id, conv2d_init_id, activation_id, beamformer_id) VALUES (12, 'BINN_OG', 6, 0, 0, 0);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 0, 'BF', 1, 1, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 1, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 2, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 5, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 1, 'BF', 16, 16, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 2, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 5, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 2, 'BF', 8, 8, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 5, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 3, 'BF', 8, 8, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 5, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 3, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 4, 'BF', 4, 4, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 5, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 3, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 4, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 5, 'BF', 2, 2, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN', 6, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 3, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 4, 'BasicConv2d', 4, 2, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 5, 'BasicConv2d', 2, 1, '[7, 5]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (7, 'BINN', 6, 'BF', 1, 1, '[-1, -1]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 0, 'BF', 1, 1, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 1, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 2, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (1, 'BINN_OG', 5, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 1, 'BF', 16, 16, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 2, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (2, 'BINN_OG', 5, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 2, 'BF', 8, 8, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 3, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (3, 'BINN_OG', 5, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 3, 'BF', 8, 8, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 4, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (4, 'BINN_OG', 5, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 3, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 4, 'BF', 4, 4, '[-1, -1]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (5, 'BINN_OG', 5, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);

INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 0, 'BasicConv2d', 1, 16, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 1, 'BasicConv2d', 16, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 2, 'BasicConv2d', 8, 8, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 3, 'BasicConv2d', 8, 4, '[5, 3]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 4, 'BasicConv2d', 4, 1, '[7, 5]', 'same', 1);
INSERT INTO architecture_cnn_bf (model_id, family, pos, type, ch_in, ch_out, kernel_json, padding, bias) VALUES (6, 'BINN_OG', 5, 'BF', 1, 1, '[-1, -1]', 'same', 1);

INSERT INTO criterion (id, type, params_json) VALUES (0, 'MSE', '{"reduction": "mean"}');
INSERT INTO criterion (id, type, params_json) VALUES (1, 'MSLE', '{"reduction": "mean"}');

INSERT INTO optimizer (id, type, params_json) VALUES (0, 'Adam', '{"amsgrad": false, "betas": [0.9, 0.999], "eps": 1e-07, "weight_decay": 0.0}');

INSERT INTO scheduler (id, type, params_json) VALUES (0, 'ReduceLROnPlateau', '{"factor": 0.5, "min_lr": 1e-06, "mode": "min", "patience": 5, "threshold": 0.0001}');

INSERT INTO hyperparameters (id, seed, n_epoch, batch_size, learning_rate) VALUES (0, 42, 100, 2, 1e-3);
INSERT INTO hyperparameters (id, seed, n_epoch, batch_size, learning_rate) VALUES (1, 42, 100, 4, 1e-3);

INSERT INTO trainloop (id, criterion_id, optimizer_id, scheduler_id, hyperparameters_id) VALUES (0, 1, 0, 0, 0);
INSERT INTO trainloop (id, criterion_id, optimizer_id, scheduler_id, hyperparameters_id) VALUES (1, 0, 0, 0, 1);

DROP TABLE IF EXISTS experiments;
DROP TABLE IF EXISTS webdataset;
DROP TABLE IF EXISTS rf_transform;
DROP TABLE IF EXISTS resize_gt;

CREATE TABLE transform_data (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE resize_gt (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE samples_organization (
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

CREATE TABLE data_size (
    id INTEGER PRIMARY KEY,
    nz INTEGER NOT NULL,
    nx INTEGER NOT NULL,
    ns INTEGER NOT NULL
);

CREATE TABLE data_type (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL CHECK (json_valid(params_json))
);

CREATE TABLE webdataset_beamformer (
    id INTEGER PRIMARY KEY,
    gt_source TEXT NOT NULL,
    data_type_id INTEGER NOT NULL,
    data_size_id INTEGER NOT NULL,
    samples_organization_id INTEGER NOT NULL,
    transform_data_id INTEGER NOT NULL,
    resize_gt_id INTEGER NOT NULL,
    FOREIGN KEY (data_type_id) REFERENCES data_type(id),
    FOREIGN KEY (data_size_id) REFERENCES data_size(id),
    FOREIGN KEY (samples_organization_id) REFERENCES samples_organization(id),
    FOREIGN KEY (transform_data_id) REFERENCES transform_data(id),
    FOREIGN KEY (resize_gt_id) REFERENCES resize_gt(id)
);

CREATE TABLE experiments (
    id INTEGER PRIMARY KEY,
    version INTEGER NOT NULL,
    webdataset_beamformer_id INTEGER NOT NULL,
    trainloop_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    commit_hash TEXT NOT NULL DEFAULT 'unknown',
    commit_msg TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (webdataset_beamformer_id) REFERENCES webdataset_beamformer(id),
    FOREIGN KEY (trainloop_id) REFERENCES trainloop(id),
    FOREIGN KEY (model_id) REFERENCES model(id)
);

INSERT INTO transform_data (id, type, params_json) VALUES (0, 'none', '{}');
INSERT INTO transform_data (id, type, params_json) VALUES (1, 'sharifzadeh', '{"eps": 1e-8}');

INSERT INTO resize_gt (id, type, params_json) VALUES (0, 'original', '{}');
INSERT INTO resize_gt (id, type, params_json) VALUES (1, 'resize', '{"new_nz": 2048, "new_nx":256, "mode": "reflect"}');

INSERT INTO samples_organization (id, seed, ratio, "order", select_mode, n_train, n_val, query, train_idxs, val_idxs) VALUES
    (0, 42, 0.9, 'CWH', 'select_idxs', 495, 55, '(RF == 1) and (nc == 128) and (name.str.slice(0, 3) != ''JHU'')', '0:12,14:19,22:51,53:57,59:70,72:86,89:90,92:98,100:101,103:105,107:120,122:129,131:159,161:165,167:188,190,192:213,215:240,242,244:251,253:263,265:269,271:272,274:275,277:307,309:312,314,316:329,331:338,340:342,344,346:365,367:371,373:384,386,388:412,414:426,428:434,436:442,444:453,455:457,460:465,467:473,476:483,485:490,492:503,506:509,511:529,531:534,536:543,545:549', '13,20:21,52,58,71,87:88,91,99,102,106,121,130,160,166,189,191,214,241,243,252,264,270,273,276,308,313,315,330,339,343,345,366,372,385,387,413,427,435,443,454,458:459,466,474:475,484,491,504:505,510,530,535,544'),
    (1, 42, 0.9, 'CWH', 'random_split', -1, -1, '(RF == 1) and (nc == 128) and (name.str.slice(0, 3) != ''JHU'') and (source == ''CUBDL'')', '-1', '-1');

INSERT INTO data_size (id, nz, nx, ns) VALUES
    (0, 2048, 256, 2300),
    (1, 2048, 256, 2800),
    (2, 1024, 256, 2300);

INSERT INTO data_type (id, type, params_json) VALUES
    (0, 'RF', '{}'),
    (1, 'RF Analitic', '{}'),
    (2, 'IQ', '{}');

INSERT INTO webdataset_beamformer (id, gt_source, data_type_id, data_size_id, samples_organization_id, transform_data_id, resize_gt_id) VALUES
    (0, 'DAS', 0, 0, 1, 1, 0),
    (1, 'DAS', 1, 0, 1, 1, 0),
    (2, 'DAS', 2, 0, 1, 1, 0),
    (3, 'DAS', 0, 2, 1, 1, 0),
    (4, 'DAS', 1, 2, 1, 1, 0),
    (5, 'DAS', 2, 2, 1, 1, 0);

INSERT INTO experiments (id, version, webdataset_beamformer_id, trainloop_id, model_id, commit_hash, commit_msg) VALUES
    (0, 1, 5, 0, 7, 'unknown', ''),
    (1, 1, 5, 0, 8, 'unknown', ''),
    (2, 1, 5, 0, 9, 'unknown', ''),
    (3, 1, 5, 0, 10, 'unknown', ''),
    (4, 1, 5, 0, 11, 'unknown', ''),
    (5, 1, 5, 0, 12, 'unknown', ''),
    (6, 1, 2, 0, 7, 'unknown', ''),
    (7, 1, 2, 0, 8, 'unknown', ''),
    (8, 1, 2, 0, 9, 'unknown', ''),
    (9, 1, 2, 0, 10, 'unknown', ''),
    (10, 1, 2, 0, 11, 'unknown', ''),
    (11, 1, 2, 0, 12, 'unknown', '');

CREATE INDEX IF NOT EXISTS idx_webdataset_beamformer_refs
    ON webdataset_beamformer(
        data_type_id,
        data_size_id,
        samples_organization_id,
        transform_data_id,
        resize_gt_id
    );

CREATE INDEX IF NOT EXISTS idx_experiments_refs
    ON experiments(webdataset_beamformer_id, trainloop_id, model_id);

COMMIT;
