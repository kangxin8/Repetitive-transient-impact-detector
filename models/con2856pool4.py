
from torch import nn
import warnings



class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=2):
        super(CNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 8, kernel_size=(28, 56)),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((4, 4)))


        self.layer5 = nn.Sequential(
            nn.Linear(8 * 4 * 4, 10))

        self.fc = nn.Linear(10, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)

        return x