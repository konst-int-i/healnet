import torch
from torch import nn

class LateFusion(nn.Module):

    def __init__(self, image_classifier: nn.Module, tabular_classifier: nn.Module, num_classes: int):
        super().__init__()
        self.image_classifier = image_classifier
        self.tabular_classifier = tabular_classifier
        # fully-connected fusion layer
        self.fc = nn.Linear(
            in_features=image_classifier.fc.out_features + tabular_classifier.fc.out_features,
            out_features=num_classes
        )

    def forward(self, images, tabular_data):
        image_output = self.image_classifier(images)
        tabular_output = self.tabular_classifier(tabular_data)
        combined_output = torch.cat((image_output, tabular_output), dim=1)
        return self.fc(combined_output)


class EarlyFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()


    def forward(self, data):
        # need to pass in MMDataSet object
        pass