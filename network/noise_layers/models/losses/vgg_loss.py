import torch
import torch.nn as nn
import torchvision
from torchvision import models


class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()
        self.vgg = models.vgg16(pretrained=True)

    def get_features(self, model, x):
        features = []
        target_layers = ["7", "17", "14", "21", "24"]
        for name, layer in model.features._modules.items():
            x = layer(x)
            if name in target_layers:
                features.append(x)
        return features

    def forward(self, x):
        return self.get_features(self.vgg, x)


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True)
        # breakpoint()
        self.vgg.load_state_dict(
            torch.load(
                "vgg_model.pth"
            )
        )
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        self.vgg = VGGModel()

        vgg_pretrained_features = self.vgg.vgg.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(8):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(8, 15):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(15, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 22):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 25):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG16().cuda()
        self.criterion = nn.L1Loss(reduction="mean")
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # self.weights = [1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0]

    def forward(self, x, y):
        # breakpoint()
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # breakpoint()
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss