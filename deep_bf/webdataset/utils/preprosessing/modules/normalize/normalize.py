from ......constants.webdataset import NormalizeOptions

from .fn import sharifzadeh_normalize

def normalize(data, normalize_mode, params):
    if normalize_mode == NormalizeOptions.NORMALIZE_NONE:
        output = data
    elif normalize_mode == NormalizeOptions.NORMALIZE_SHARIFZADEH:
        output = sharifzadeh_normalize(data, **params)
    else:
        output = data

    return output
