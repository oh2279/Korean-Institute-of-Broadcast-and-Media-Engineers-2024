import torch
import torch.nn as nn
import sys

cfg = {
    'vgg5':  [64,'M', 128, 128, 'M'],
    'vgg9':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, input_size=32, num_class=100, in_channels = 3):
        super(VGG, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.n_maps = cfg[vgg_name][-2]
        self.fc1 = self._make_fc_layers()
        self.classifier = nn.Linear(self.n_maps*self.input_size*self.input_size, num_class)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        #out = self.fc1(out)
        out = self.classifier(out)
        return out
    
    def _make_fc_layers(self):
        layers = []
        layers += [nn.Linear(self.n_maps*self.input_size*self.input_size, 1024),
                   nn.BatchNorm1d(1024),
                   nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def VGG5(input_size, num_class):
    return VGG('VGG5', input_size, num_class)
def VGG9(input_size, num_class):
    return VGG('VGG9', input_size, num_class)

def VGG11(input_size, num_class):
    return VGG('VGG11', input_size, num_class)

def test():
    #net = VGG('VGG11', input_size=96,num_class=10)
    net = VGG('VGG11', input_size=32,num_class=10)
    print(net)
    #x = torch.randn(2, 3, 96, 96)
    x = torch.randn(4, 3, 32, 32)
    y = net(x)
    print(y.size())
#test()