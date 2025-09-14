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

class SimpleMLP(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(5, 64),  # 입력 5차원
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.classifier_han_yeol = nn.Linear(64, num_class)
        self.classifier_huh_sil = nn.Linear(64, num_class)
        self.regressor_han = nn.Linear(64, 1)
        self.regressor_yeol = nn.Linear(64, 1)
        self.regressor_huh = nn.Linear(64, 1)
        self.regressor_sil = nn.Linear(64, 1)

    def forward(self, x):  # x: (batch, 1, 5) or (batch, 5)
        x = x.view(x.size(0), -1)  # (batch, 5)
        shared = self.shared_fc(x)
        return (
            self.classifier_han_yeol(shared),
            self.classifier_huh_sil(shared),
            self.regressor_han(shared),
            self.regressor_yeol(shared),
            self.regressor_huh(shared),
            self.regressor_sil(shared)
        )


class VGG9ANN(nn.Module):
    def __init__(self, vgg_name, input_size=36000, num_class=2, in_channels=5):
        super(VGG9ANN, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.n_maps = cfg[vgg_name][-2]
        
        # Shared FC layer
        self.shared_fc = nn.Sequential(
            nn.Linear(self.n_maps*self.input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Multiple classifiers and regressors (following SNN structure)
        self.classifier_han_yeol = nn.Linear(512, num_class)
        self.classifier_huh_sil = nn.Linear(512, num_class)
        
        self.regressor_han = nn.Linear(512, 1)
        self.regressor_yeol = nn.Linear(512, 1)
        self.regressor_huh = nn.Linear(512, 1)
        self.regressor_sil = nn.Linear(512, 1)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        
        # Shared feature extraction
        shared_features = self.shared_fc(out)
        
        # Multiple outputs
        x1 = self.classifier_han_yeol(shared_features)
        x2 = self.classifier_huh_sil(shared_features)
        x3 = self.regressor_han(shared_features)
        x4 = self.regressor_yeol(shared_features)
        x5 = self.regressor_huh(shared_features)
        x6 = self.regressor_sil(shared_features)
        
        return x1, x2, x3, x4, x5, x6
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm1d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def VGG5(input_size=36000, num_class=2):
    return VGG9ANN('vgg5', input_size, num_class)

def VGG9(input_size=36000, num_class=2):
    return VGG9ANN('vgg9', input_size, num_class)

def VGG11(input_size=36000, num_class=2):
    return VGG9ANN('vgg11', input_size, num_class)

def test():
    print("=== Testing VGG9ANN with Multiple Outputs ===")
    net = VGG9(input_size=36000, num_class=2)
    print(net)
    
    # Test input: [batch=4, channels=5, sequence=36000]
    x = torch.randn(4, 5, 36000)
    print(f"Input shape: {x.shape}")
    
    outputs = net(x)
    print(f"Number of outputs: {len(outputs)}")
    
    output_names = ['classifier_han_yeol', 'classifier_huh_sil', 'regressor_han', 
                   'regressor_yeol', 'regressor_huh', 'regressor_sil']
    
    for i, (output, name) in enumerate(zip(outputs, output_names)):
        print(f"{name} shape: {output.shape}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")

if __name__ == '__main__':
    test()