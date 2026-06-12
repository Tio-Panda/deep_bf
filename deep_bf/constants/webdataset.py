from enum import StrEnum

class SplitOptions(StrEnum):
    SPLIT_MODE_SELECT_IDX = "select_idxs"
    SPLIT_MODE_RANDOM = "random_split"

class NormalizeOptions(StrEnum):
    NORMALIZE_NONE = "none"
    NORMALIZE_SHARIFZADEH = "sharifzadeh"
