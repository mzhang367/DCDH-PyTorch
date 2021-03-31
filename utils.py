# coding: utf-8
import torch
import torchvision.transforms.transforms as transforms
from torch.utils.data import Dataset

import numpy as np
import pickle
import os
import sys
import errno
import os.path as osp


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x = np.clip(x, -1, 1)
    return x


def get_data(dataset, train=False):
    x = []
    y = []

    if dataset == "facescrub":
        root_dir_train = "./FaceScrub/train"
        root_dir_test = "./FaceScrub/test"

    else:
        root_dir_train = "./YouTube/train"
        root_dir_test = "./YouTube/test"


    if train:
        dir = root_dir_train
    else:
        dir = root_dir_test
    for file in os.listdir(dir):
        filepath = os.path.join(dir, file)
        with open(filepath, "rb") as f:
            data, label = pickle.load(f, encoding='bytes')
            x.extend(data)
            y.extend(label)

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=int)
    x = deprocess_image(x)
    return x, y


transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, dataset, transform, train):
        super(Dataset, self).__init__()
        self.transform = transform
        self.train = train
        if self.train:
            self.train_x, self.train_y = get_data(dataset=dataset, train=True)
            self.train_x = self.train_x.reshape((len(self.train_x), 3, 32, 32))
            self.train_x = self.train_x.transpose((0, 2, 3, 1))
            self.train_y = self.train_y.reshape(len(self.train_y), 1)
        else:
            self.test_x, self.test_y = get_data(dataset=dataset, train=False)
            self.test_x = self.test_x.reshape((len(self.test_x), 3, 32, 32))
            self.test_x = self.test_x.transpose((0, 2, 3, 1))
            self.test_y = self.test_y.reshape(len(self.test_y), 1)

    def __getitem__(self, index):
        if self.train:
            imgs, labels = self.train_x[index], self.train_y[index]
        else:
            imgs, labels = self.test_x[index], self.test_y[index]

        if self.transform is not None:
            imgs = self.transform(imgs)
        imgs = imgs.type(torch.FloatTensor)
        labels = torch.from_numpy(labels)
        return imgs, labels

    def __len__(self):
        if self.train:
            return len(self.train_x)
        else:
            return len(self.test_x)


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()# first reset

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


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def compute_result(dataloader, net, device):

    """
    return hashing codes of data with shape (N, len_bits) and its labels (N, )
    """
    hash_codes = []
    label = []
    for i, (imgs, cls, *_) in enumerate(dataloader):
        imgs, cls = imgs.to(device), cls.to(device)
        hash_values = net(imgs)
        hash_codes.append(hash_values.data)
        label.append(cls)

    hash_codes = torch.cat(hash_codes)
    B = torch.where(hash_codes > 0.0, torch.tensor([1.0]).cuda(), torch.tensor([-1.0]).cuda())

    return B, torch.cat(label)


def compute_topK(trn_binary, tst_binary, trn_label, tst_label, device, top_list):

    """
    compute mean precision of returned top-k results based on Hamming ranking
    """

    top_p = torch.Tensor(tst_binary.size(0), len(top_list)).to(device)

    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        for j, top in enumerate(top_list):
            top_result = query_result[:top]
            top_correct = (query_label == trn_label[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p[i, j] = 1.0*N_top/top

    top_pres = top_p.mean(dim=0).cpu().numpy()

    return top_pres


def compute_mAP(trn_binary, tst_binary, trn_label, tst_label, device):

    AP = []
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        N = torch.sum(correct)
        Ns = torch.arange(1, N+1).float().to(device)
        index = (torch.nonzero(correct, as_tuple=False)+1)[:, 0].float()
        AP.append(torch.mean(Ns / index))

    mAP = torch.mean(torch.Tensor(AP))
    return mAP


def evaluate_recall_pre(train_labels, test_labels, train_bits, test_bits, device):

    """
    Precision and Recall are based on 2 hamming distances
    """

    num_test = test_bits.size(0)

    hammRadius = 2
    q = train_bits.shape[1]

    precisions = torch.zeros((num_test)).to(device)
    recalls = torch.zeros((num_test)).to(device)

    for j in range(num_test):

        distH = 0.5 * (q - torch.mm(test_bits[j, :].unsqueeze(0), train_bits.transpose(0, 1))).transpose(0, 1)
        Ret = torch.le(distH, hammRadius + 1e-6).squeeze_()
        cateTrainTest = torch.eq(train_labels, test_labels[j]).squeeze_()
        retrieved_relevant_pairs = cateTrainTest & Ret

        retrieved_relevant_num = torch.nonzero(retrieved_relevant_pairs).numel()    # true positive

        retrieved_num = torch.nonzero(Ret).numel()  # true pos. + false pos.

        relevant_num = torch.nonzero(cateTrainTest).numel()   # true pos. + false neg.

        if retrieved_num:
            precisions[j] = retrieved_relevant_num / retrieved_num

        else:
            precisions[j] = 0.0

        if relevant_num:
            recalls[j] = retrieved_relevant_num / relevant_num

        else:
            recalls[j] = 0.0

    p = torch.mean(precisions)
    r = torch.mean(recalls)
    return p, r


def evaluate_pre_curve(trn_binary, tst_binary, trn_label, tst_label, ranking_list):

    """
    Calculate points to plot the precision curve w.r.t. Hamming ranking given a series of ranking positions
    args:
        ranking_list: the ranking positions to be evaluated
    """

    top_p = np.ndarray((tst_binary.shape[0], len(ranking_list)))
    top_r = np.ndarray((tst_binary.shape[0], len(ranking_list)))

    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        cateTrainTest = torch.eq(trn_label, query_label).squeeze_()
        revelant_num = torch.nonzero(cateTrainTest).numel()
        for k, top in enumerate(ranking_list):
            top_result = query_result[:top]
            top_correct = (query_label == trn_label[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p[i, k] = 1.0*N_top/top
            top_r[i, k] = 1.0*N_top/revelant_num

    top_pre = np.mean(top_p, axis=0)
    top_recall = np.mean(top_r, axis=0)

    return top_pre, top_recall


def EncodingOnehot(target, nclasses):
    target_onehot = torch.Tensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.cpu().view(-1, 1), 1)
    return target_onehot
