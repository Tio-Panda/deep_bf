import torch


def apply_apodization(tofc_data, apod):
    nc = apod.shape[0]
    spatial_shape = apod.shape[1:]

    if tofc_data.dim() == 4 and tofc_data.shape[1] == nc and tofc_data.shape[2:] == spatial_shape:
        return tofc_data * apod.unsqueeze(0)

    if (
        tofc_data.dim() == 5
        and tofc_data.shape[1] == nc
        and tofc_data.shape[2:4] == spatial_shape
        and tofc_data.shape[-1] == 2
    ):
        return tofc_data * apod.unsqueeze(0).unsqueeze(-1)

    if tofc_data.dim() == 5 and tofc_data.shape[2] == nc and tofc_data.shape[3:] == spatial_shape:
        return tofc_data * apod.unsqueeze(0).unsqueeze(0)

    raise ValueError(
        f"Unsupported tofc_data shape {tuple(tofc_data.shape)} for apod shape {tuple(apod.shape)}"
    )

def get_window(distance, aperture, kind="boxcar"):
    eps = 1e-10
    rel_dist = distance / (aperture + eps)
    
    mask = distance <= aperture

    if kind == "boxcar":
        w = mask.float()
        
    elif kind == "hanning":
        val = 0.5 + 0.5 * torch.cos(torch.pi * rel_dist)
        w = torch.where(mask, val, torch.zeros_like(distance))
        
    elif kind == "hamming":
        val = 0.53836 + 0.46164 * torch.cos(torch.pi * rel_dist)
        w = torch.where(mask, val, torch.zeros_like(distance))
        
    elif kind.startswith("tukey"):
        try:
            roll = float(kind[5:]) / 100.0
        except ValueError:
            roll = 0.5
            
        flat_region = distance < aperture * (1 - roll)
        taper_val = 0.5 * (1 + torch.cos((torch.pi / (roll + eps)) * (rel_dist - 1 + roll)))
        
        w = torch.where(flat_region, 
                        torch.ones_like(distance), 
                        torch.where(mask, taper_val, torch.zeros_like(distance)))
    else:
        raise ValueError(f"Ventana '{kind}' no reconocida en PyTorch.")

    return w
    
def dynamic_receive_aperture(Z, X, probe_geometry, f_num=1.5, window="boxcar", device="cuda", dtype=torch.float32):
    probe = torch.from_numpy(probe_geometry[:, 0]).to(device=device, dtype=dtype)
    distance = torch.abs(probe[:, None, None] - X[None, :, :])  # [n_elements, nz, nx]
    aperture = Z[None, :, :] / (2.0 * f_num) # [1, nz, nx]

    apod = get_window(distance, aperture, kind=window) # [n_elements, nz, nx]

    return apod
