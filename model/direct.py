from model.layers import SeqToANNContainer,LIFSpike,Layer,add_dimention,FCLayer
import torch
import torch.nn as nn

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

    def forward(self, inp_1):
        inp_1 = self.stem_conv(inp_1)
        
        if len(inp_1.shape) == 4: # [B, C, H, W] -> [T, B, C, H, W]
            spike_inp_1 = add_dimention(inp_1, self.T)
        else: spike_inp_1  = inp_1
        
        x = self.features(spike_inp_1)
        return x
        # x = torch.flatten(x, 2)
        # x = self.classifier(x)

        # return x

