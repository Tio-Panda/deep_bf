from enum import StrEnum


class PWDataType(StrEnum):
    RF = "RF"
    IQ_COMPLEX_DEMOD = "IQComplexDemod"
    IQ_COMPLEX = "IQComplex"
    IQ_SPLIT = "IQSplit"


class BeamformerType(StrEnum):
    DAS = "DAS"
    FDMAS = "F-DMAS"
    MV = "MV"
    CF = "CF"
    IMAP = "iMAP"
    SR1 = "SparseRegularization"
    SR2 = "SparseRegularization2"


class ResamplerType(StrEnum):
    GRID_SAMPLE = "GridSample"
    LINEAR_INTERPOLATION = "LinearInterpolation"
    INTEGER_FRACTIONAL_FIR = "IntegerDelayFractionalFIR"


class CompooundingType(StrEnum):
    CPWC_SUM = "CoherentPlane-WaveCompoundingWithSum"
    CPWC_MEAN = "CoherentPlane-WaveCompoundingWithMean"
    ANGULAR_CPWC_SHORT_PULSE = "CoherentPlane-WaveCompoundingAngularApodizationShortPulse"
    ANGULAR_CPWC_PARAXIAL = "CoherentPlane-WaveCompoundingAngularApodizationParaxial"
    CPWC_EDT = "CoherentPlane-WaveCompoundingWithEDT"
