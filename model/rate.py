import random
from model.layers import *
from snntorch.spikegen import rate

def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

# torch.le(a, b) -> b element가 a element보다 작으면 false

class VGG5SNN(nn.Module):
    def __init__(self, num_cls, time_step,in_channels=3,input_size=32):
        super(VGG5SNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        ap_pool = APLayer(2)
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
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(self.n_maps*W*W,self.num_cls))
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, inp_1):
        # spike_inp_1_list = []

        # for t in range(self.T):
        #     spike_inp_1_list.append(PoissonGen(inp_1))

        # spike_inp_1 = torch.stack(spike_inp_1_list, dim=0)
        #print(inp_1.shape)
        
        spike_inp_1 = rate(inp_1, num_steps = self.T)
        #print(x.shape)
        x = self.features(spike_inp_1)

        x = torch.flatten(x, 2)
        #x = self.fc1(x)
        #x = self.fc2(x)
        x = self.classifier(x)
        #print(x.mean(0).shape)
        return x.mean(0)


