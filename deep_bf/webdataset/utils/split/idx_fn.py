import numpy as np
import pandas as pd

def encode_idxs(names, selected_names):
    """
    names: pd.Series (ej: df["name"])
    selected_names: iterable de nombres seleccionados (asumidos existentes)
    return: string compactado de índices base 0, rangos inclusivos
            ej: '0:299,303,423'
    """
    selected = set(selected_names)
    mask = names.isin(selected).to_numpy()
    idx_list = [i for i, ok in enumerate(mask) if ok]
    if not idx_list:
        return ""
    parts = []
    start = prev = idx_list[0]
    for x in idx_list[1:]:
        if x == prev + 1:
            prev = x
        else:
            parts.append(f"{start}:{prev}" if start != prev else str(start))
            start = prev = x
    parts.append(f"{start}:{prev}" if start != prev else str(start))
    return ",".join(parts)

def decode_idxs(names, idxs):
    """
    names: pd.Series (ej: df["name"])
    idxs: string compactado base 0, rangos inclusivos (ej: '0:2,5,7:8')
    return: list[str] con nombres decodificados
    """
    idxs = idxs.strip()
    if not idxs:
        return []
    positions = []
    for part in idxs.split(","):
        part = part.strip()
        if ":" in part:
            a, b = map(int, part.split(":", 1))
            positions.extend(range(a, b + 1))
        else:
            positions.append(int(part))
    return names.iloc[positions].tolist()
