from model.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import encoding
import snntorch
from snntorch.spikegen import latency

class VGG5SNN(nn.Module):
    def __init__(self, num_cls, time_step,in_channels=3,input_size=32):
        super(VGG5SNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.num_cls = num_cls
        self.T = time_step
        self.in_channels = in_channels
        self.input_size = input_size
        self.act = LIFSpike()
        self.n_maps = 128

        self.stem_conv = nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1)
        self.stem_bn = nn.BatchNorm2d(self.in_channels)

        self.features = nn.Sequential(
            Layer(self.in_channels,64,3,1,1),
            pool,
            Layer(64,128,3,1,1),
            Layer(128,128,3,1,1),
            pool,
        )
        W = int(self.input_size/2/2)

        self.classifier = SeqToANNContainer(nn.Linear(self.n_maps*W*W,self.num_cls))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    def forward(self, x): 
        
        # Normalize input to [0, 1] range
        x = (x - x.min()) / (x.max() - x.min())
        #print(x.shape, x.max(), x.min())
        x = latency(data = x, num_steps = self.T, normalize=True,linear=True)
        x = self.features(x)
         
        x = torch.flatten(x, 2)
        x = self.classifier(x)

        return x.mean(0)
    
def add_dimention(inp, T):
    return inp.unsqueeze(0).repeat(T, 1, 1, 1, 1)