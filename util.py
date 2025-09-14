import torch
import torch.nn.functional as F

class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad
    
def get_mean_and_std_1d(dataset):
    '''Compute the mean and std value of 1D dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(1)
    std = torch.zeros(1)
   # logger.info('==> Computing mean and std for 1D data..')
    for idx, data in enumerate(dataloader):
        print(idx)
        inputs = data['image']
        print(idx, inputs.shape)
        mean += inputs.mean()
        std += inputs.std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    
    return mean.item(), std.item()

def adjust_learning_rate(optimizer, cur_epoch, max_epoch):
    if (
        cur_epoch == (max_epoch * 0.5)
        or cur_epoch == (max_epoch * 0.7)
        or cur_epoch == (max_epoch * 0.9)
    ):
        for param_group in optimizer.param_groups:
            param_group["lr"] /= 10

def accuracy(outp, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = outp.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count