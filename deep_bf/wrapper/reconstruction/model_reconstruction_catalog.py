from __future__ import annotations

from collections import defaultdict


class ModelReconstructionCatalog:
    def __init__(self):
        self._items = []
        self._all_ids = set()
        self._index_name = defaultdict(set)
        self._index_nz = defaultdict(set)
        self._index_nx = defaultdict(set)
        self._index_ns = defaultdict(set)
        self._index_preprocessing_type = defaultdict(set)
        self._index_preprocessing_id = defaultdict(set)
        self._index_beamformer_type = defaultdict(set)
        self._index_compounding_type = defaultdict(set)
        self._index_model_pack_id = defaultdict(set)
        self._index_model_family = defaultdict(set)
        self._index_experiment_id = defaultdict(set)

    def add(self, model_reconstruction):
        idx = len(self._items)
        self._items.append(model_reconstruction)
        self._all_ids.add(idx)

        if getattr(model_reconstruction, "name", None) is not None:
            self._index_name[model_reconstruction.name].add(idx)

        ds_c = model_reconstruction.data_size_config
        if ds_c is not None:
            if getattr(ds_c, "nz", None) is not None:
                self._index_nz[ds_c.nz].add(idx)
            if getattr(ds_c, "nx", None) is not None:
                self._index_nx[ds_c.nx].add(idx)
            if getattr(ds_c, "ns", None) is not None:
                self._index_ns[ds_c.ns].add(idx)

        dp_c = model_reconstruction.data_preprocessing_config
        if dp_c is not None:
            if getattr(dp_c, "type", None) is not None:
                self._index_preprocessing_type[dp_c.type].add(idx)
            if getattr(dp_c, "id", None) is not None:
                self._index_preprocessing_id[dp_c.id].add(idx)

        exp_c = model_reconstruction.experiment
        has_exp_beamformer_type = False
        has_exp_compounding_type = False
        if exp_c is not None:
            exp_wdb_p = getattr(exp_c, "webdataset_beamformer_pack", None)
            if exp_wdb_p is not None:
                exp_bf_setup = getattr(exp_wdb_p, "beamformer_setup", None)
                if exp_bf_setup is not None:
                    exp_bf_c = getattr(exp_bf_setup, "beamformer_config", None)
                    if exp_bf_c is not None and getattr(exp_bf_c, "type", None) is not None:
                        self._index_beamformer_type[exp_bf_c.type].add(idx)
                        has_exp_beamformer_type = True

                    exp_compounding_c = getattr(exp_bf_setup, "compounding_config", None)
                    if exp_compounding_c is not None and getattr(exp_compounding_c, "type", None) is not None:
                        self._index_compounding_type[exp_compounding_c.type].add(idx)
                        has_exp_compounding_type = True

        mp_c = model_reconstruction.model_pack
        if mp_c is not None:
            if getattr(mp_c, "id", None) is not None:
                self._index_model_pack_id[mp_c.id].add(idx)
            if getattr(mp_c, "family", None) is not None:
                self._index_model_family[mp_c.family].add(idx)

            # Fallback when experiment-based indexing data is unavailable.
            bf_setup = getattr(mp_c, "beamformer_setup", None)
            if bf_setup is not None:
                if not has_exp_beamformer_type:
                    bf_c = getattr(bf_setup, "beamformer_config", None)
                    if bf_c is not None and getattr(bf_c, "type", None) is not None:
                        self._index_beamformer_type[bf_c.type].add(idx)

                if not has_exp_compounding_type:
                    compounding_c = getattr(bf_setup, "compounding_config", None)
                    if compounding_c is not None and getattr(compounding_c, "type", None) is not None:
                        self._index_compounding_type[compounding_c.type].add(idx)

        if exp_c is not None and getattr(exp_c, "id", None) is not None:
            self._index_experiment_id[exp_c.id].add(idx)

    def all(self):
        return list(self._items)

    def query(
        self,
        *,
        name=None,
        nz=None,
        nx=None,
        ns=None,
        preprocessing_type=None,
        preprocessing_id=None,
        beamformer_type=None,
        compounding_type=None,
        model_pack_id=None,
        model_family=None,
        experiment_id=None,
    ):
        candidates = set(self._all_ids)

        if name is not None:
            candidates &= self._index_name.get(name, set())
        if nz is not None:
            candidates &= self._index_nz.get(nz, set())
        if nx is not None:
            candidates &= self._index_nx.get(nx, set())
        if ns is not None:
            candidates &= self._index_ns.get(ns, set())
        if preprocessing_type is not None:
            candidates &= self._index_preprocessing_type.get(preprocessing_type, set())
        if preprocessing_id is not None:
            candidates &= self._index_preprocessing_id.get(preprocessing_id, set())
        if beamformer_type is not None:
            candidates &= self._index_beamformer_type.get(beamformer_type, set())
        if compounding_type is not None:
            candidates &= self._index_compounding_type.get(compounding_type, set())
        if model_pack_id is not None:
            candidates &= self._index_model_pack_id.get(model_pack_id, set())
        if model_family is not None:
            candidates &= self._index_model_family.get(model_family, set())
        if experiment_id is not None:
            candidates &= self._index_experiment_id.get(experiment_id, set())

        ordered_ids = sorted(candidates)
        return [self._items[idx] for idx in ordered_ids]
