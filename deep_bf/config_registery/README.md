# config_registery

Isolated SQLite-backed config module for deep_bf.

## Run migrations

```bash
python -m deep_bf.config_registery.db.migrate
```

Custom DB path:

```bash
python -m deep_bf.config_registery.db.migrate --db "/path/to/config_registery.db"
```

## Use the center

```python
from deep_bf.config_registery import ConfigRegisteryCenter

with ConfigRegisteryCenter("/path/to/config_registery.db") as cc:
    experiment = cc.get_experiment(0)
    tables = cc.list_tables()
    experiments_df = cc.show_table("experiment")
```

## Add API rules

- `add_*` for `*Config` entities receive a config object.
- `add_*_setup`, `add_*_pack`, and `add_experiment` receive foreign keys as `int`.
- Incoming config IDs are ignored on insert; SQLite generates IDs.
- `ArchitectureCnnBfConfig` is the exception: it has no `id` and uses composite key `(model_id, family, pos)`.
- Every successful `add_*` writes one delta seed with only the inserted row.
- If delta export fails, insert remains successful and a warning is emitted.

## Get API rules

- Primitive config getters follow `get_*` names that mirror `add_*`:
  - `get_conv2d_init`, `get_activation`, `get_architecture`
  - `get_criterion`, `get_optimizer`, `get_scheduler`, `get_hyperparameters`
  - `get_data_type`, `get_resampler`, `get_beamformer`, `get_compounding`, `get_apod`
  - `get_data_size`, `get_data_preprocessing`, `get_samples_organization`
- `get_architecture(family, model_id)` returns all architecture rows for that model ordered by `pos`.
- Composite getters remain available:
  `get_beamformer_setup`, `get_trainloop_setup`, `get_model_pack`,
  `get_webdataset_beamformer_pack`, `get_experiment`.

Auto-export config:

```python
ConfigRegisteryCenter(
    "/path/to/config_registery.db",
    auto_export_delta_on_add=True,
    delta_seeds_out_dir=None,
)
```

Example:

```python
from deep_bf.config_registery import ConfigRegisteryCenter
from deep_bf.config_registery.entities import (
    ActivationConfig,
    BeamformerConfig,
    DataTypeConfig,
    ResamplerConfig,
)

with ConfigRegisteryCenter("/path/to/config_registery.db") as cc:
    act = cc.add_activation(ActivationConfig(id=-1, type="LeakyReLU", params={"negative_slope": 0.01}))
    dt = cc.add_data_type(DataTypeConfig(id=-1, type="RF", params={}))
    rs = cc.add_resampler(ResamplerConfig(id=-1, type="LinearInterpolation", params={}))
    bf = cc.add_beamformer(BeamformerConfig(id=-1, type="DAS", params={}))

    setup = cc.add_beamformer_setup(
        data_type_config_id=dt.id,
        beamformer_config_id=bf.id,
        resampler_config_id=rs.id,
        compounding_config_id=0,
        apod_config_id=0,
    )
```

## Export full seed

```bash
python -m deep_bf.config_registery.db.export_full_seed \
  --db "/path/to/config_registery.db" \
  --out "/path/to/full_seed.sql"
```

## Export one delta seed manually

Single primary key table:

```bash
python -m deep_bf.config_registery.db.export_delta_seed \
  --db "/path/to/config_registery.db" \
  --table beamformer_config \
  --pk 10
```

Composite primary key table:

```bash
python -m deep_bf.config_registery.db.export_delta_seed \
  --db "/path/to/config_registery.db" \
  --table architecture_cnn_bf_config \
  --pk "model_id=2,family=BINN,pos=1"
```

## Rebuild DB

```bash
python -m deep_bf.config_registery.db.rebuild \
  --db "/path/to/rebuilt_config_registery.db" \
  --drop-existing \
  --full-seed "/path/to/full_seed.sql" \
  --deltas-dir "/path/to/delta_seeds"
```

Behavior:
- Migrations run first.
- Full seed runs next when provided and found.
- Delta seeds run last in lexicographic filename order.
- Skip full seed or deltas with empty values (`--full-seed ""`, `--deltas-dir ""`).
