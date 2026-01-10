import torch

def compute_meshgrid(pw, nz, nx, z_lims=None, x_lims=None, device='cuda', dtype=torch.float32):
    """
    Compute meshgrid using PyTorch for GPU acceleration.

    Args:
        pw: Plane wave object
        nz: Number of points in z direction
        nx: Number of points in x direction
        z_lims: Z limits [min, max]. If None, uses pw.zlims
        x_lims: X limits [min, max]. If None, uses [-pw.aperture_width/2, pw.aperture_width/2]
        dtype: PyTorch dtype (default: torch.float16)
        device: Device to create tensors on (default: 'cuda')

    Returns:
        Z, X: Meshgrid tensors of shape [nz, nx] on the specified device
    """
    if z_lims is None: z_lims = pw.zlims
    if x_lims is None: x_lims = [-pw.aperture_width/2, pw.aperture_width/2]

    z = torch.linspace(z_lims[0], z_lims[-1], nz, dtype=dtype, device=device)
    x = torch.linspace(x_lims[0], x_lims[-1], nx, dtype=dtype, device=device)
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


def compute_d_tx(pw, Z, X, angle_indices=None, device='cuda', dtype=torch.float32):
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

    if angle_indices is None:
        angle_indices = list(range(len(pw.angles)))
    elif isinstance(angle_indices, int):
        angle_indices = [angle_indices]
    else:
        angle_indices = list(angle_indices)

    angles = torch.from_numpy(pw.angles[angle_indices]).to(dtype=dtype, device=device)
    t0 = torch.from_numpy(pw.t0[angle_indices]).to(dtype=dtype, device=device)

    angles_exp = angles.view(-1, 1, 1)
    d_tx = X * torch.sin(angles_exp) + Z * torch.cos(angles_exp) # [n_angles, nz, nx]

    return d_tx, t0


def compute_samples(pw, d_tx, d_rx, t0, decimation=1):
    """
    Compute sample indices using PyTorch for GPU acceleration.

    Args:
        pw: Plane wave object
        d_tx: Transmit distances of shape [n_angles, nz, nx]
        d_rx: Receive distances of shape [n_elements, nz, nx]
        t0: Time offset of shape [n_angles]
        decimation: Decimation factor (default: 1)

    Returns:
        samples: Sample indices of shape [n_angles, n_elements, nz, nx]
    """
    # d_tx: [n_angles, nz, nx] -> [n_angles, 1, nz, nx]
    # d_rx: [n_elements, nz, nx] -> [1, n_elements, nz, nx]
    # t0: [n_angles] -> [n_angles, 1, 1, 1]
    total_delay = (d_tx.unsqueeze(1) + d_rx.unsqueeze(0)) / pw.c0 - t0.view(-1, 1, 1, 1)
    samples = total_delay * (pw.fs / decimation) # [n_angles, n_elements, nz, nx]
    return samples
