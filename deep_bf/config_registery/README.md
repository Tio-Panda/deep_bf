# config_registery

Isolated SQLite-backed config module for deep_bf.

## Run migrations

```bash
python -m deep_bf.config_registery.db.migrate
```

You can also target a custom DB path:

```bash
python -m deep_bf.config_registery.db.migrate --db "/path/to/config_registery.db"
```

## Use the center

```python
from deep_bf.config_registery import ConfigRegisteryCenter

cc = ConfigRegisteryCenter("/path/to/config_registery.db")
experiment_config = cc.get_experiment_config(0)
experiment_packing = cc.get_experiment_packing(0)
webdataset_beamformer_config = cc.get_webdataset_beamformer_config(0)
webdataset_beamformer_packing = cc.get_webdataset_beamformer_packing(0)
model_config = cc.get_model_config(0)
model_packing = cc.get_model_packing(0)
trainloop_config = cc.get_trainloop_config(0)
trainloop_packing = cc.get_trainloop_packing(0)
tables = cc.list_tables()
experiments_df = cc.show_table("experiments")
cc.close()
```

Notes:
- `params` payloads are stored as JSON text in SQLite.
- `beamformer` stores `id`, `type`, `resampler_id`, and `params_json`.
- `apod` configs are stored independently and are not nested in `BeamformerPacking`.
- `samples_organization` stores `query`, `train_idxs`, and `val_idxs` as raw strings.
- `experiments` references `webdataset_beamformer_id` and includes `commit_hash` and `commit_msg` defaults.
