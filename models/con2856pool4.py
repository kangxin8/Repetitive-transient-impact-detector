
from torch import nn
import warnings



class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=2):
        super(CNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 8, kernel_size=(28, 56)),  # 16, 26 ,26
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((4, 4)))  # 32, 12,12     (24-2) /2 +1)


        self.layer5 = nn.Sequential(

            nn.Linear(8 * 4 * 4, 10))

        self.fc = nn.Linear(10, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)

        return x

class revised_con2856pool4(nn.Module):
    def __init__(self, num_classes):
        super(revised_con2856pool4, self).__init__()
        # 获取特征提取器
        model_base = CNN()  # 创建模型对象
        checkpoint = torch.load(
            r'F:\OneDrive - hnu.edu.cn\项目\project_code\first_paper_transient_detection\checkpoint\con7pool2_CWRUSlice_0528-201234\9-0.9988-best_model.pth')
        model_base.load_state_dict(checkpoint)  # 向模型中填入模型参数
        model_crop = nn.Sequential(*list(model_base.children())[:-2])
        self.feature_extract = nn.Sequential(*model_crop.children())
        for param in self.feature_extract.parameters():
            param.requires_grad = False
        # 定义分类器
        self.classifier = nn.Linear(in_features=8*4*4, out_features=num_classes)

    def forward(self, x):
        x = self.feature_extract(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
