import numpy as np
from scipy.signal import hilbert
from ...constants.bf import PWDataType

def get_bmode(data, mode, vmin=-60, vmax=0, eps=1e-10):
    if mode == PWDataType.RF:
        env = np.abs(hilbert(data, axis=0))
        #env = np.abs(data)
    elif mode in (PWDataType.IQ_COMPLEX, PWDataType.IQ_COMPLEX_DEMOD):
        env = np.abs(data)
    elif mode == PWDataType.IQ_SPLIT:
        env = np.linalg.norm(data, axis=-1)
    else:
        env = np.linalg.norm(data, axis=-1)  # sqrt(I^2 + Q^2)
    
    # TODO: Analizar bien cual es la problematica del b-mode (la simu da una envolvente por los 1e-24)
    # env = np.asarray(env, dtype=np.float32)
    # env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalizacion lineal (clave para señales de muy baja amplitud)
    max_env = float(np.max(env))
    env = env / max_env
    eps_val = max(float(eps), np.finfo(env.dtype).eps)
    env = np.maximum(env, eps_val)

    # Piso numerico sin mover el maximo por encima de 0 dB
    b_mode = 20.0 * np.log10(env)

    # ref = np.percentile(env, 99.9)
    # b_mode = 20 * np.log10(env / ref)

    b_mode = np.clip(b_mode, vmin, vmax)

    # b_mode = 20 * np.log10(env + eps)
    # b_mode -= np.amax(b_mode)
    # b_mode = np.clip(b_mode, vmin, vmax)

    return b_mode
