import torch
from ..beamformers import RFGridSampleDAS, RFInterDAS
from ..beamformers import compute_meshgrid, compute_d_tx, compute_d_rx
from ..beamformers import dynamic_receive_aperture

from ..wrapper.reconstruction import Reconstruction
from ..wrapper.metadata import ClassicMetadata

# Los modelos de IA tendran en su nombre, toda su configuracion.

class BenchmarkWrapper:
    def __init__(self, nz, nx, pw, angles_idx, bf_names, f_num, window, device="cuda", dtype=torch.float32):
        self.nz = nz
        self.nx = nx
        self.angles_idx = angles_idx
        self.f_num = f_num
        self.window = window

        self.device = device
        self.dtype = dtype

        self.nz, self.nx = nz, nx
        self.c0 = pw.c0
        self.fs = pw.fs
        self.rf = torch.from_numpy(pw.data[angles_idx]).to(device=device, dtype=dtype) # [n_angles, n_elements, n_samples]
        self.t0 = torch.from_numpy(pw.t0[angles_idx]).to(device=device, dtype=dtype)

        self.beamformers = []

        for name in bf_names:
            match name:
                case "RFGridSampleDAS":
                    bf_name = "RFGridSampleDAS"
                    bf = RFGridSampleDAS(batch_size=12, device=device, dtype=dtype)
                    metadata = ClassicMetadata("DAS", "Grid Sample", self.f_num, self.window)
                case "RFInterDAS":
                    bf_name = "RFInterDAS"
                    bf = (RFInterDAS(batch_size=1, device=device, dtype=dtype))
                    metadata = ClassicMetadata("DAS", "Grid Sample", self.f_num, self.window)
                case _:
                    bf_name = "RFGridSampleDAS"
                    bf = RFGridSampleDAS(batch_size=12, device=device, dtype=dtype)
                    metadata = ClassicMetadata("DAS", "Grid Sample", self.f_num, self.window)
                
            self.beamformers.append((bf_name, bf, metadata))


        self.Z, self.X = compute_meshgrid(pw, nz, nx, device=device, dtype=dtype)
        self.d_tx, self.t0 = compute_d_tx(pw, self.Z, self.X, device=device, dtype=dtype)
        self.d_rx = compute_d_rx(pw, self.Z, self.X, device=device, dtype=dtype)

        self.apod = dynamic_receive_aperture(self.Z, self.X, pw.probe_geometry, f_num, window, device=device, dtype=dtype)

    def compute_reconstructions(self):
        out = {}

        for i, (bf_name, bf, metadata) in enumerate(self.beamformers):
            data = bf(self.rf, self.t0, self.d_tx, self.d_rx, self.fs, self.c0, self.apod)
            
            reconstruction = Reconstruction(data, self.Z, self.X, metadata)
            out[bf_name] = reconstruction

        return out
