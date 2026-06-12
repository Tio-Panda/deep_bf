from __future__ import annotations

from collections import defaultdict

class ReconstructionCatalog:
    def __init__(self):
        self._items = []
        self._all_ids = set()
        self._index_name = defaultdict(set)
        self._index_nz = defaultdict(set)
        self._index_nx = defaultdict(set)
        self._index_bf_type = defaultdict(set)
        self._index_apod_type = defaultdict(set)
        self._index_resampler_type = defaultdict(set)
        self._index_compounding_type = defaultdict(set)

    def add(self, reconstruction):
        idx = len(self._items)
        self._items.append(reconstruction)
        self._all_ids.add(idx)

        if getattr(reconstruction, "name", None) is not None:
            self._index_name[reconstruction.name].add(idx)

        ds_c = reconstruction.data_size_config
        if ds_c is not None:
            self._index_nz[ds_c.nz].add(idx)
            self._index_nx[ds_c.nx].add(idx)

        bf_c = reconstruction.beamformer_config
        if bf_c is not None:
            self._index_bf_type[bf_c.type].add(idx)

        apod_c = reconstruction.apod_config
        if apod_c is not None:
            self._index_apod_type[apod_c.type].add(idx)

        resampler_c = reconstruction.resampler_config
        if resampler_c is not None:
            self._index_resampler_type[resampler_c.type].add(idx)

        compounding_c = reconstruction.compounding_config
        if compounding_c is not None:
            self._index_compounding_type[compounding_c.type].add(idx)

    def all(self):
        return list(self._items)

    def query(
        self,
        *,
        name=None,
        nz=None,
        nx=None,
        bf_type=None,
        apod_type=None,
        resampler_type=None,
        compounding_type=None,
    ):
        candidates = set(self._all_ids)

        if name is not None:
            candidates &= self._index_name.get(name, set())
        if nz is not None:
            candidates &= self._index_nz.get(nz, set())
        if nx is not None:
            candidates &= self._index_nx.get(nx, set())
        if bf_type is not None:
            candidates &= self._index_bf_type.get(bf_type, set())
        if apod_type is not None:
            candidates &= self._index_apod_type.get(apod_type, set())
        if resampler_type is not None:
            candidates &= self._index_resampler_type.get(resampler_type, set())
        if compounding_type is not None:
            candidates &= self._index_compounding_type.get(compounding_type, set())

        ordered_ids = sorted(candidates)
        return [self._items[idx] for idx in ordered_ids]
