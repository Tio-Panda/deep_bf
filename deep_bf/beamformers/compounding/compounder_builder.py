from ...config_registery import CompoundingConfig
from ...constants.bf import CompooundingType
from .methods.cpwc import CPWC
from .methods.angularCPWC import AngularCPWCParaxial, AngularCPWCShortPulse
from .methods.cpwcEdt import CPWCEDT


def compounder_builder(config: CompoundingConfig, pw=None):
    type = config.type
    params = config.params

    if type == CompooundingType.CPWC_SUM or type == CompooundingType.CPWC_MEAN:
        cmp = CPWC(type)
    elif type == CompooundingType.ANGULAR_CPWC_SHORT_PULSE:
        cmp = AngularCPWCShortPulse(pw=pw, **params)
    elif type == CompooundingType.ANGULAR_CPWC_PARAXIAL:
        cmp = AngularCPWCParaxial(pw=pw, **params)
    elif type == CompooundingType.CPWC_EDT:
        cmp = CPWCEDT(**params)
    else:
        cmp = CPWC(type)

    return cmp
