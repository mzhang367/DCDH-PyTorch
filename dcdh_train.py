from utils import *
from model import DFHNet
import torch.optim as optim
from datetime import datetime
import sys
import argparse
from Loss import DualClasswiseLoss
from torch.utils.data import DataLoader
import time
import torch.backends.cudnn as cudnn
from torchvision import datasets
from InceptionRes_ft_pytorch.inception_resnet_v1 import InceptionResnetV1



parser = argparse.ArgumentParser(description='PyTorch Implementation of Paper: Deep Center-based Dual-constrained Hashing(DCDH).')
parser.add_argument('--lr1', default=0.005, type=float, help='learning rate of backbone network')
parser.add_argument('--lr2', default=0.005, type=float, help='learning rate of loss layer')

parser.add_argument('--save', type=str, help='path to saving model')
parser.add_argument('--dataset', type=str, default='facescrub', help='should be one of {facescrub, youtube, vgg}')
parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
parser.add_argument('--len', type=int, default=48, help='length of hashing codes,  should be one of {12, 24, 36, 48}')

# hyper params.
parser.add_argument('--sigma', default=0.25, type=float, help='class gap of ClasswiseLoss')
parser.add_argument('--inner_param', default=0.1, type=float, help='balance weight on two constraints')
parser.add_argument('--lamda', default=1, type=float, help='regularization on regression')
parser.add_argument('--eta', default=0.01, type=float, help='quantization weight')

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True


if args.dataset in ['facescrub', 'youtube']:

    EPOCHS = 700
    transform_tensor = transforms.Compose([
        transforms.ToTensor()])
    trainset = MyDataset(args.dataset, transform=transform_tensor, train=True)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)
    testset = MyDataset(args.dataset, transform=transform_tensor, train=False)
    testloader = DataLoader(testset, batch_size=args.bs, shuffle=False)
    net = torch.nn.DataParallel(DFHNet(args.len)).to(device)
    classes = len(np.unique(trainset.train_y))

else:
    EPOCHS = 100
    trainPaths = "./vggface2/train"
    testPaths = "./vggface2/test"
    cropped_size = 160
    Normalize = transforms.Normalize((0.5141, 0.4074, 0.3588), (1, 1, 1))

    transform_train = transforms.Compose([
        transforms.Resize(cropped_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Normalize,
    ])

    transform_validation = transforms.Compose([
        transforms.Resize(cropped_size),
        transforms.ToTensor(),
        Normalize,
     ])

    trainset = datasets.ImageFolder(root=trainPaths, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=6)
    testset = datasets.ImageFolder(root=testPaths, transform=transform_validation)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=6)
    classes = len(trainset.classes)
    Inception = InceptionResnetV1(pretrained="vggface2", fc=True, num_bits=args.len)
    net = torch.nn.DataParallel(Inception).to(device)


def adjust_learning_rate(optimizer, epoch):

    """Sets the learning rate to the initial LR decayed by 0.5 every 100 epochs"""
    lr = []
    lr.append(args.lr1 * (0.5 ** (epoch // 100)))
    lr.append(args.lr2 * (0.5 ** (epoch // 100)))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr[i]
    return lr


def train(EPOCHS):

    print('==> Preparing training data..')

    if args.dataset in ['facescrub', 'youtube']:

        print("number of training images: ", len(trainset.train_y))
        print("number of classes: ", classes)
        print("number of test images: ", len(testset.test_y))
        print("number of training iterations per epoch:", len(trainloader))

    else:
        print("number of training images: ", len(trainset))
        print("number of classes: ", classes)
        print("number of test images: ", len(testset))
        print("number of training iterations per epoch:", len(trainloader))

    criterion = DualClasswiseLoss(num_classes=classes, inner_param=args.inner_param, sigma=args.sigma, feat_dim=args.len, use_gpu=True)

    best_epoch = 0
    best_loss = 1e4
    if args.dataset in ['facescrub', 'youtube']:
        optimizer = optim.Adam([
            {'params': net.module.parameters(), 'weight_decay': 1e-4, 'lr': args.lr1, 'amsgrad': True},
            {'params':  criterion.parameters(), 'weight_decay': 1e-4, 'lr': args.lr2}
        ])
    else:
        optimizer = optim.SGD([
            {'params': net.module.parameters(), 'weight_decay': 5e-4},
            {'params':  criterion.parameters(), 'weight_decay': 5e-4}
        ], lr=args.lr, momentum=0.9)

    since = time.time()
    for epoch in range(EPOCHS):
        print('==> Epoch: %d' % (epoch + 1))
        net.train()
        dcdh_loss = AverageMeter()
        adjust_learning_rate(optimizer, epoch)
        # epoch_start = time.time()
        for batch_id, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            hash_bits = net(imgs)
            loss_dual = criterion(hash_bits, labels)################ difference between imageloader and custom loader
            hash_binary = torch.sign(hash_bits)
            batchY = EncodingOnehot(labels, classes).cuda()
            W = torch.mm(torch.inverse(torch.mm(torch.transpose(batchY, 0, 1), batchY) + args.lamda * torch.eye(batchY.size(1)).cuda()),
            torch.mm(torch.transpose(batchY, 0, 1), hash_binary))    # Update W

            batchB = torch.sign(torch.mm(batchY, W) + args.eta * hash_bits)  # Update B

            loss_vertex = (hash_bits - batchB).pow(2).sum() / len(imgs)
            loss_h = loss_dual + args.eta * loss_vertex

            dcdh_loss.update(loss_h.item(), len(imgs))
            loss_h.backward()
            optimizer.step()

        print("[epoch: %d]\t[hashing loss: %.3f ]" % (epoch+1, dcdh_loss.avg))

        if (epoch+1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                centers_trained = torch.sign(criterion.centers.data).cuda()
                trainB, train_labels = compute_result(trainloader, net, device, centers_trained)
                testB, test_labels = compute_result(testloader, net, device, centers_trained)
                mAP = compute_mAP(trainB, testB, train_labels, test_labels, device)
                print('[Evaluate Phase] Epoch: %d\t mAP: %.2f%%' % (epoch+1, 100. * float(mAP)))

        if dcdh_loss.avg < best_loss:
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save({'backbone': net.module.state_dict(),
                        'centers': criterion.state_dict()}, './checkpoint/%s' % args.save)
            best_loss = dcdh_loss.avg
            best_epoch = epoch

        if (epoch - best_epoch) > EPOCHS // 4:
            print("Training terminated at epoch %d" %(epoch + 1))
            break

    time_elapsed = time.time() - since
    print("Training Completed in {:.0f}min {:.0f}s with best loss in epoch {}".format(time_elapsed // 60, time_elapsed % 60, best_epoch + 1))
    print("Model saved as %s" % args.save)



if __name__ == '__main__':

    if not os.path.isdir('log'):
        os.mkdir('log')
    save_dir = './log'

    assert args.save
    sys.stdout = Logger(os.path.join(save_dir,
    str(args.len) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
    print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n #Epoch: %d\n"
    %(args.dataset, args.len, args.bs, args.lr1, EPOCHS))
    print("HyperParams:\nsigma: %.3f\t inner_param: %.4f\t eta: %.4f\t lamda: %.4f" % (args.sigma, args.inner_param, args.eta, args.lamda))
    train(EPOCHS)
