import time
import numpy as np
import torch
import torch.nn.functional as F

def compute_meshgrid(pw, nz, nx, z_lims=None, x_lims=None, nyquist=True, device='cuda', dtype=torch.float32):
    """
    Compute meshgrid using PyTorch for GPU acceleration.

    Args:
        pw: Plane wave object
        nz: Number of points in z direction
        nx: Number of points in x direction
        z_lims: Z limits [min, max]. If None, uses pw.zlims
        x_lims: X limits [min, max]. If None, uses [-pw.aperture_width/2, pw.aperture_width/2]
        dtype: PyTorch dtype (default: torch.float32)
        device: Device to create tensors on (default: 'cuda')

    Returns:
        Z, X: Meshgrid tensors of shape [nz, nx] on the specified device
    """
    # if z_lims is None: z_lims = pw.zlims
    # # if x_lims is None: x_lims = pw.xlims
    # if x_lims is None: x_lims = [-pw.aperture_width/2, pw.aperture_width/2]

    z_lims = pw.zlims
    x_lims = pw.xlims

    if not nyquist:
        z = torch.linspace(z_lims[0], z_lims[-1], nz, dtype=dtype, device=device)
        x = torch.linspace(x_lims[0], x_lims[-1], nx, dtype=dtype, device=device)
    else:
        z0, z1 = z_lims
        x0, x1 = x_lims

        c0 = float(pw.c0)
        fc = float(pw.fc)
        lam = c0 / (fc + 1e-10)

        dz = lam / 4.0
        dx = lam / 3.0

        z = torch.arange(z0, z1 + 0.5*dz, dz, dtype=dtype, device=device)
        x = torch.arange(x0, x1 + 0.5*dx, dx, dtype=dtype, device=device)


    Z, X = torch.meshgrid(z, x, indexing="ij") # Both [nz, nx]
    return Z, X



def compute_d_rx(pw, Z, X, device='cuda', dtype=torch.float32):
    """
    Compute receive distances using PyTorch for GPU acceleration.

    Args:
        pw: Plane wave object
        Z, X: Meshgrid tensors of shape [nz, nx]
        dtype: PyTorch dtype (default: torch.float16)
        device: Device to create tensors on (default: 'cuda')

    Returns:
        d_rx: Receive distances of shape [n_elements, nz, nx]
    """
    ele_pos = torch.from_numpy(pw.probe_geometry[:, 0]).to(dtype=dtype, device=device)
    ele_pos = ele_pos.view(-1, 1, 1)
    d_rx = torch.sqrt((X - ele_pos)**2 + Z**2) # [n_elements, nz, nx]
    return d_rx


def compute_d_tx(pw, Z, X, angles_idx=None, device='cuda', dtype=torch.float32):
    """
    Compute transmit distances using PyTorch for GPU acceleration.

    Args:
        pw: Plane wave object
        Z, X: Meshgrid tensors of shape [nz, nx]
        angle_indices: Indices of angles to compute. Can be:
                      - None (default): computes all angles
                      - Single integer: computes only that angle
                      - List/array of integers: computes only specified angles
                      Examples: None, 37, [0, 37, 74], np.arange(25, 50)
        dtype: PyTorch dtype (default: torch.float16)
        device: Device to create tensors on (default: 'cuda')

    Returns:
        d_tx: Transmit distances of shape [n_angles, nz, nx]
        t0: Time offset of shape [n_angles]
    """

    if angles_idx is None:
        angles_idx = np.arange(pw.n_angles)

    angles = pw.angles[angles_idx]
    t0 = pw.t0[angles_idx]

    angles = torch.from_numpy(angles).to(dtype=dtype, device=device)
    t0 = torch.from_numpy(t0).to(dtype=dtype, device=device)

    angles_exp = angles.view(-1, 1, 1)
    d_tx = X * torch.sin(angles_exp) + Z * torch.cos(angles_exp) # [n_angles, nz, nx]

    return d_tx, t0

# TODO: Para implementar con un parametro "demod=True" Para utilizar pw.fc en vez de pw.fs
def compute_samples_idx_by_angles(pw, Z, X, d_rx, angles_idx, device="cuda", dtype=torch.float32):
    d_tx, t0 = compute_d_tx(pw, Z, X, angles_idx=angles_idx, device=device, dtype=dtype)
    samples = pw.fs * (((d_tx.unsqueeze(1) + d_rx.unsqueeze(0)) / pw.c0) + t0.view(-1, 1, 1, 1))
    # samples = samples.clamp(0.0, float(ns-1))

    #TODO: Ver si se puede dejar sample pero con un c0 personalizado. Pero puede ser para mas para adelante

    return samples
