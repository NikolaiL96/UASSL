import torch


def grad_clip_hook_(model):
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -2.0, 2.0))
