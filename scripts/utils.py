from __future__ import annotations

import torch.nn as nn
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop


def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
    if norm_type == "BatchNorm":
        return nn.BatchNorm2d(num_features, eps=eps)
    if norm_type == "GroupNorm":
        return nn.GroupNorm(num_groups, num_features, eps=eps)
    if norm_type == "InstanceNorm":
        return nn.InstanceNorm2d(
            num_features,
            eps=eps,
            affine=True,
            track_running_stats=True,
        )
    raise Exception(f"Unknown Norm Function : {norm_type}")


def get_optimizer(params, cfg):
    if cfg.optimizer == "SGD":
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "Adadelta":
        return Adadelta(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "Adagrad":
        return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "Adam":
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "RMSprop":
        return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    raise Exception(f"Unknown optimizer : {cfg.optimizer}")


def tensor2numpy(input_tensor):
    return input_tensor.cpu().detach().numpy()