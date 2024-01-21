import models
import os
from networks import *
import torch
from distributions import Probabilistic_Layer
from utils.utils import _get_model_name, _split_name
from utils import load_dataset



METHOD_CLS = 'method_cls'
DEFAULT_OPTIONS = 'default_options'

MODEL_CONFIG = {
    'BarlowTwins': {
        METHOD_CLS: models.BarlowTwins,
        DEFAULT_OPTIONS: {"projector_hidden": (2048, 2048, 1024)}
    },
    'SimCLR': {
        METHOD_CLS: models.SimCLR,
        DEFAULT_OPTIONS: {"projector_hidden": (2048, 2048, 256),
                          "loss": "InfoNCE",
                          "temperature": 0.05,
                          "lambda_reg": 0.01}
    },
}


class ModelFactory:
    def __init__(self, model_id, network_id, in_channels=3, distribution_type='none', model_options=None):
        if model_id not in MODEL_CONFIG.keys():
            raise ValueError('Unknown model_id')

        self.model_cls = MODEL_CONFIG[model_id][METHOD_CLS]
        self.model_options = model_options if model_options is not None else MODEL_CONFIG[
            model_id][DEFAULT_OPTIONS]
        self.distribution_type = distribution_type
        self.network_id = network_id
        self.in_channels = in_channels

    def build_model(self):
        return self._construct_model()

    def load_from_checkpoint(self, checkpoint_path, device='cuda', finetune_prob_layer=""):
        model = self._construct_model()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        params = checkpoint["model"]
        msg = model.load_state_dict(params, strict=False)
        print(msg)
        # We use a different Prob Layer
        if finetune_prob_layer:
            model.backbone_net.fc = Probabilistic_Layer(
                finetune_prob_layer, model.repre_dim)
        model.to(device)
        return model, checkpoint

    def _construct_model(self):
        backbone = self._build_network(self.network_id, self.in_channels)
        model = self.model_cls(
            backbone,
            **self.model_options,
            distribution_type=self.distribution_type,
        )
        return model

    def _build_network(self, network, in_channels):
        # call different resnets
        if network == 'resnet18':
            backbone = cifar_resnet18(in_channels)
        elif network == 'resnet50':
            backbone = cifar_resnet50(in_channels)
        elif network == "uncertainty_net":
            backbone = UncertaintyNet(in_channels, self.distribution_type)
        else:
            raise ValueError('Unknown network id')
        return backbone
