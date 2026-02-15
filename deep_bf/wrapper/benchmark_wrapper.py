import torch
from ..beamformers import DASGridSample, DASManual, FDMAS, MVB
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

        self.c0 = pw.c0
        self.fs = pw.fs
        self.f0 = pw.fc
        self.rf = torch.from_numpy(pw.data[angles_idx]).to(device=device, dtype=dtype) # [n_angles, n_elements, n_samples]
        self.t0 = torch.from_numpy(pw.t0[angles_idx]).to(device=device, dtype=dtype) # [n_angles]

        self.beamformers = []

        for name in bf_names:
            match name:
                case "DASGridSample":
                    bf_name = "DASGridSample"
                    bf = DASGridSample(batch_size=12, device=device, dtype=dtype)
                    metadata = ClassicMetadata("DAS", "Grid Sample", self.f_num, self.window)
                case "DASManual":
                    bf_name = "DASManual"
                    bf = DASManual(batch_size=1, device=device, dtype=dtype)
                    metadata = ClassicMetadata("DAS", "Grid Sample", self.f_num, self.window)
                case "FDMAS":
                    bf_name = "FDMAS"
                    bf = FDMAS(BW=0.8, for_dmas=False, batch_size=8, device=device, dtype=dtype)
                    metadata = ClassicMetadata("fDMAS", "Grid Sample", self.f_num, self.window)
                case "MVB":
                    bf_name = "MVB"
                    bf = MVB(
                         device=self.device,
                         dtype=self.dtype)
                    metadata = ClassicMetadata("MVB", "Grid Sample", self.f_num, self.window)
                case _:
                    bf_name = "DASGridSample"
                    bf = DASGridSample(batch_size=12, device=device, dtype=dtype)
                    metadata = ClassicMetadata("DAS", "Grid Sample", self.f_num, self.window)
                
            self.beamformers.append((bf_name, bf, metadata))


        self.Z, self.X = compute_meshgrid(pw, nz, nx, device=device, dtype=dtype)
        self.d_tx, self.t0 = compute_d_tx(pw, self.Z, self.X, device=device, dtype=dtype)
        self.d_rx = compute_d_rx(pw, self.Z, self.X, device=device, dtype=dtype)

        self.apod = dynamic_receive_aperture(self.Z, self.X, pw.probe_geometry, f_num, window, device=device, dtype=dtype)

    def compute_reconstructions(self):
        out = {}

        for bf_name, bf, metadata in self.beamformers:
            data = bf(self.rf, self.t0, self.d_tx, self.d_rx, self.fs, self.f0, self.c0, self.apod)
            
            reconstruction = Reconstruction(data, self.Z, self.X, metadata)
            out[bf_name] = reconstruction

        return out

    def get_samples_idx_by_angle(self, angle):
        import torch.nn.functional as F
        grid_template = None
        _rf = self.rf[angle]
        _d_tx = self.d_tx[angle]
        _t0 = self.t0[angle]
        
        nz, nx = _d_tx.shape
        n_channels, n_samples = _rf.shape
        norm_factor = 2.0 / (n_samples - 1)

        samples = self.fs * (((_d_tx.unsqueeze(0) + self.d_rx) / self.c0) + _t0)

        return samples
        # x = _rf.view(n_channels, 1, 1, n_samples)
        # 
        # if grid_template is None or grid_template.shape[:3] != (n_channels, nz, nx):
        #     grid_template = torch.empty(n_channels, nz, nx, 2, device=self.device, dtype=self.dtype)
        # 
        # grid_template[..., 1] = 0
        # grid_template[..., 0] = samples * norm_factor - 1.0
        # samples_idx = F.grid_sample(
        #     x, 
        #     grid_template,
        #     mode="bilinear",
        #     padding_mode="zeros",
        #     align_corners=True
        # )
        #
        # return samples_idx.squeeze()


