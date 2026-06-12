from ..config_registery import (
    TrainLoopSetup,
    CriterionConfig, 
    OptimizerConfig,
    SchedulerConfig
)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .loss.msle import MSLELoss

CRITERION_MSE = "MSE"
CRITERION_MSLE = "MSLE"

def set_criterion(config: CriterionConfig):
    criterion_type = config.type
    params = config.params

    if criterion_type == CRITERION_MSE:
        criterion = nn.MSELoss(**params)
    elif criterion_type == CRITERION_MSLE:
        criterion = MSLELoss(**params)
    else:
        criterion = nn.MSELoss(**params)

    return criterion


OPTIMIZER_ADAM = "Adam"

def set_optimizer(model: nn.Module, config: OptimizerConfig):
    optimizer_type = config.type
    params = config.params
    params["params"] = model.parameters()

    if optimizer_type == OPTIMIZER_ADAM:
        optimizer = optim.Adam(**params)
    else:
        optimizer = optim.Adam(**params)

    return optimizer


SCHEDULER_NONE = "None"
SCHEDULER_REDUCELRONPLATEAU = "ReduceLROnPlateau"


def set_scheduler(optimizer, config: SchedulerConfig):
    scheduler_type = config.type
    params = config.params
    params["optimizer"] = optimizer

    if scheduler_type == SCHEDULER_REDUCELRONPLATEAU:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(**params)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(**params)

    return scheduler


def set_train_strategy(model: nn.Module, config: TrainLoopSetup):
    cC = config.criterion_config
    oC = config.optimizer_config
    sC = config.scheduler_config

    criterion = set_criterion(cC)
    optimizer = set_optimizer(model, oC)

    scheduler = None
    if sC.type != SCHEDULER_NONE:
        scheduler = set_scheduler(optimizer, sC)

    return criterion, optimizer, scheduler
