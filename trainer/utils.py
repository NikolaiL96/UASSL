import torch


def grad_clip_hook_(model, clip=2., clip_type="Hookfc"):
    # No clipping needed if clip<0.
    if clip <= 0.:
        return

    # Clipping function
    clip_fn = lambda grad: torch.clamp(grad, -clip, clip)

    # Determine which parameters to apply the gradient clipping
    parameters = model.fc.parameters() if "fc" in clip_type else model.parameters()

    # Apply backward_hook
    for p in parameters:
        p.register_hook(clip_fn)


def get_params_(fine_tune, model, reduced_lr, lr, logger, lr_fc=6e-3):
    # Define set of trainable parameters
    if fine_tune:
        # When finetune the probabilistic layer
        params = model.backbone_net.fc.parameters()

    elif (reduced_lr is True) and (model.backbone_net.name == "UncertaintyNet"):
        params = [{'params': [k[1] for k in model.named_parameters() if 'kappa' in k[0]], 'lr': lr_fc},
                  {'params': [k[1] for k in model.named_parameters() if 'kappa' not in k[0]]}]
        logger.info(f"Learning rate of {lr} for the backbone and {lr_fc} for KappaModel.")

    elif ("resnet" in model.backbone_net.name) and (reduced_lr is True):
        params = [
            {'params': [k[1] for k in model.named_parameters() if 'Probabilistic_Layer' in k[0]], 'lr': lr_fc},
            {'params': [k[1] for k in model.named_parameters() if 'Probabilistic_Layer' not in k[0]]}]
        logger.info(f"Learning rate of {lr} for the backbone and {lr_fc} for fc.")

    else:
        params = model.parameters()

    return params
