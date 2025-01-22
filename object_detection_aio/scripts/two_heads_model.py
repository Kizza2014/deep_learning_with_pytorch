import torch.nn as nn
import torchvision
import torch

class TwoHeadModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # resnet18 backbone
        self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        # remove fc of original resnet18
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # classification head
        self.classifier = nn.Linear(in_features=num_features, out_features=num_classes)

        # regression head
        self.bbox_regressor = nn.Linear(in_features=num_features, out_features=4)

    def forward(self, x):
        x = self.backbone(x)
        classifier_logits = self.classifier(x)
        bbox_coords = torch.sigmoid(self.bbox_regressor(x))
        return classifier_logits, bbox_coords