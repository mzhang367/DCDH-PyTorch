import torch
import torch.nn as nn
import torch.nn.functional as F


def Log(x):
    """
    Log trick for numerical stability
    """

    lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.tensor([0.]).cuda())

    return lt


class DualClasswiseLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, inner_param, sigma=0.25, use_gpu=True):
        super(DualClasswiseLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.sigma = sigma
        self.inner_param = inner_param
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels):
        """
        Args:
            x: shape of (batch_size, feat_dim).
            labels: shape of (batch_size, ) or (batch_size, 1)
        """

        #   compute L_1 with single constraint.
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        dist_div = torch.exp(-0.5*self.sigma*distmat)/(torch.exp(-0.5*self.sigma*distmat).sum(dim=1, keepdim=True) + 1e-6)
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.view(-1, 1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist_log = torch.log(dist_div+1e-6) * mask.float()
        loss = -dist_log.sum() / batch_size

        #   compute L_2 with inner constraint on class centers.
        centers_norm = F.normalize(self.centers, dim=1)
        theta_x = 0.5 * self.feat_dim * centers_norm.mm(centers_norm.t())
        mask = torch.eye(self.num_classes, self.num_classes).bool().cuda()
        theta_x.masked_fill_(mask, 0)
        loss_iner = Log(theta_x).sum() / (self.num_classes*(self.num_classes-1))

        loss_full = loss + self.inner_param * loss_iner
        return loss_full
