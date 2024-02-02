
import torch.nn as nn

class Supervised(nn.Module):
    def __init__(self, backbone_net: nn.Module, num_classes: int = 10, distribution_type: str = None,):
        super().__init__()

        self.backbone_net = backbone_net
        self.rep_dim = self.backbone_net.fc.in_features
        self.backbone_net.fc = nn.Linear(self.rep_dim, num_classes)

        # Verbose
        print(f"We initialize Supervised training with {self.rep_dim} dimensions.")


    def forward(self, x):
        return self.backbone_net(x)

