import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        #print(y_seq.view(y_shape).shape)
        return y_seq.view(y_shape)

class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        #norm_layer = tdBN()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x
    

class FCLayer(nn.Module):
    def __init__(self,in_plane,out_plane):
        super(FCLayer, self).__init__()
        #norm_layer = tdBN()
        self.fwd = SeqToANNContainer(
            nn.Linear(in_plane,out_plane),
            nn.BatchNorm1d(out_plane)
        )
        #self.bn = tdBN(num_features=out_plane, alpha=1.0, v_th=1.0)
        self.act = LIFSpike()

    def forward(self,x):
        #print(x.shape)
        x = self.fwd(x)
        #print(x.shape)
        # x = self.bn(x)
        x = self.act(x)
        #print(x.shape)
        return x

class APLayer(nn.Module):
    def __init__(self,kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class Surrogate_BP_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad, None


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.99, gama=1.0):
        super(LIFSpike, self).__init__()
        # self.act = ZIF.apply
        self.act = Surrogate_BP_Function.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        # self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem * self.tau + x[t, :, ...]
            spike = self.act(mem - self.thresh)
            # spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            # mem = spike*self.thresh + (1-spike)*mem # soft reset
            mem = mem - spike*self.thresh # soft reset 0430
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=0)

class LIFSpike_hard(nn.Module):
    def __init__(self, thresh=1.0, tau=0.99, gama=1.0):
        super(LIFSpike_hard, self).__init__()
        # self.act = ZIF.apply
        self.act = Surrogate_BP_Function.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        # self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem * self.tau + x[t, :, ...]
            spike = self.act(mem - self.thresh)
            # spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            # mem = spike*self.thresh + (1-spike)*mem # soft reset
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=0)
# class LIFSpike(nn.Module):
#     def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
#         super(LIFSpike, self).__init__()
#         self.act = ZIF.apply
#         # self.k = 10
#         # self.act = F.sigmoid
#         self.thresh = thresh
#         self.tau = tau
#         self.gama = gama

#     def forward(self, x):
#         mem = 0
#         spike_pot = []
#         T = x.shape[1]
#         for t in range(T):
#             mem = mem * self.tau + x[:, t, ...]
#             spike = self.act(mem - self.thresh, self.gama)
#             # spike = self.act((mem - self.thresh)*self.k)
#             mem = (1 - spike) * mem
#             spike_pot.append(spike)
#         return torch.stack(spike_pot, dim=1)

def add_dimention(x, T):
    x.unsqueeze_(0)
    x = x.repeat(T, 1, 1, 1, 1)
    return x


# ----- For ResNet19 code -----


# class tdLayer(nn.Module):
#     def __init__(self, layer, bn=None):
#         super(tdLayer, self).__init__()
#         self.layer = SeqToANNContainer(layer)
#         self.bn = bn

#     def forward(self, x):
#         x_ = self.layer(x)
#         if self.bn is not None:
#             x_ = self.bn(x_)
#         return x_

class tdLayer_direct(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(tdLayer_direct, self).__init__()
        # norm_layer = tdBN()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
        )
        self.bn = tdBatchNorm(channel=out_plane)
        self.act = LIFSpike()

    def forward(self,x, direct):
        x = self.fwd(x)
        x = self.bn(x)
        if direct is True:
            x = self.act(x)

        # T, B, C, H, W = x.shape
        # print(x.shape)
        # print(x[4:].shape)
        # print(x.sum())
        
        # spike_counts = x.sum() / x.numel() *T
    

        # return x, spike_counts
        return x

class tdLayer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(tdLayer, self).__init__()
        # norm_layer = tdBN()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
        )
        self.bn = tdBatchNorm(channel=out_plane)
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.bn(x)
        x = self.act(x)
 
        # T, B, C, H, W = x.shape
        # print(x.shape)
        # print(x[4:].shape)
        # print(x.sum())
        
        # spike_counts = x.sum() / x.numel() *T
    

        # return x, spike_counts
        return x

class tdLayer_hard(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(tdLayer_hard, self).__init__()
        # norm_layer = tdBN()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
        )
        self.bn = tdBatchNorm(channel=out_plane)
        self.act = LIFSpike_hard()

    def forward(self,x):
        x = self.fwd(x)
        x = self.bn(x)
        x = self.act(x)

        T, B, C, H, W = x.shape
        # print(x.shape)
        # print(x[4:].shape)
        # print(x.sum())
        
        spike_counts = x.sum() / x.numel() *T
    

        return x, spike_counts
        return x

class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, channel):
        super(tdBatchNorm, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        self.weight.data.mul_(0.5)

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        out = super().forward(x.reshape(T * B, *spatial_dims))
        BT, *spatial_dims = out.shape
        out = out.view(T, B, *spatial_dims).contiguous()
        return out

# LIFSpike = LIF

def add_dimention(x, T):
    # x.unsqueeze_(0)
    # x = x.repeat(T, 1, 1, 1, 1)
    x = torch.stack([x]*T, dim=0)
    return x