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

    def forward(self, x, labels, centroids=None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: Noted labels are with shape (batch_size, 1).
        """

        #   compute L_1 with single constraint.
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        dist_div = torch.exp(-0.5*self.sigma*distmat)/(torch.exp(-0.5*self.sigma*distmat).sum(dim=1, keepdim=True) + 1e-8)
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist_log = torch.log(dist_div+1e-8) * mask.float()
        loss = -dist_log.sum() / batch_size

        #   compute L_2 with inner constraint on class centers.
        centers_norm = F.normalize(self.centers, dim=1)

        theta_x = 0.5 * self.feat_dim * centers_norm.mm(centers_norm.t())

        #theta_x = self.centers.mm(self.centers.t()) / 2draw_plot_pr.py
        mask = torch.eye(self.num_classes, self.num_classes).bool().cuda()
        theta_x.masked_fill_(mask, 0)
        loss_iner = Log(theta_x).sum() / (self.num_classes*(self.num_classes-1))

        loss_full = self.inner_param * loss_iner + loss

        return loss_full


class MetricDualClasswiseLoss(nn.Module):

    """
    impose margin factor on the corresponding class centers; non explicit Gaussian distribution form.
    """

    def __init__(self, num_classes, feat_dim, intra_param, alpha=0.2, use_gpu=True):
        super(MetricDualClasswiseLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.alpha = alpha
        self.inner_param = intra_param
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels, centroids=None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: Noted labels are with shape (batch_size, 1).
        """

        #   compute L_1 with single constraint.
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())  # shape of (bs * num_bit)
        distmat_margin = (1 + self.alpha) * distmat
        one_hot = torch.zeros(batch_size, self.num_classes)  # first create all zero matrix
        one_hot = one_hot.cuda() if self.use_gpu else one_hot
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        # classes = torch.arange(self.num_classes).long()

        output = one_hot * (distmat - distmat_margin) - distmat

        loss = F.cross_entropy(output, labels.squeeze(), reduction='mean')

        #   compute L_2 with inner constraint on class centers.
        C_matrix = [self.centers[label, :].view(1, self.feat_dim) for label in labels.squeeze().tolist()]   # consider using all hashing codes in a batch
        C_matrix = torch.cat(C_matrix)
        loss_intra = 0.5 * (x - C_matrix).pow(2).sum() / batch_size

        loss_full = self.inner_param * loss_intra + loss

        return loss_full

