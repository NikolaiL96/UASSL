import torch


def grad_clip_hook_(model, clip=2.):
    if clip != 0.:
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
