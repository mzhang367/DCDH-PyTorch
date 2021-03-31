from utils import *
from model import DFHNet, SphereNet_hashing
import torch.optim as optim
from datetime import datetime
import sys
import argparse
from loss import DualClasswiseLoss
from torch.utils.data import DataLoader
import time
import torch.backends.cudnn as cudnn
from torchvision import datasets

parser = argparse.ArgumentParser(description='PyTorch Implementation of Paper: Deep Center-based Dual-constrained Hashing(DCDH).')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate of backbone network and hashing loss layer')
# recommend using lr = 0.01 for VggFace2 dataset
parser.add_argument('--save', type=str, help='path to saving model')
parser.add_argument('--dataset', type=str, default='facescrub', help='should be one of {facescrub, youtube, vggface2}')
parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
parser.add_argument('--len', type=int, default=48, help='length of hashing codes,  should be one of {12, 24, 36, 48}')

# hyper params.
parser.add_argument('--sigma', default=0.25, type=float, help='class gap of ClasswiseLoss')
parser.add_argument('--inner_param', default=0.1, type=float, help='balance weight on two constraints')
parser.add_argument('--eta', default=0.01, type=float, help='quantization weight')

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True

class adjust_lr:
    """
    create a class instance;
    multiply DECAY every STEP epochs
    """
    def __init__(self, step, decay):
        self.step = step
        self.decay = decay

    def adjust(self, optimizer, epoch):
        lr = args.lr * (self.decay ** (epoch // self.step))
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
        return lr


if args.dataset in ['facescrub', 'youtube']:

    EPOCHS = 600
    transform_tensor = transforms.Compose([
        transforms.ToTensor()])
    trainset = MyDataset(args.dataset, transform=transform_tensor, train=True)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
    testset = MyDataset(args.dataset, transform=transform_tensor, train=False)
    testloader = DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)
    net = DFHNet(args.len).to(device)
    classes = len(np.unique(trainset.train_y))
    scheduler = adjust_lr(100, 0.5)

else:
    EPOCHS = 150
    trainPaths = "./vggface2/train"
    testPaths = "./vggface2/test"

    transform_train = transforms.Compose([
                    transforms.Resize(120),
                    transforms.RandomCrop(112),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
                    transforms.Resize(120),
                    transforms.CenterCrop(112),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.ImageFolder(root=trainPaths, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
    testset = datasets.ImageFolder(root=testPaths, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)
    classes = len(trainset.classes)
    net = SphereNet_hashing(num_layers=20, hashing_bits=args.len).to(device)
    scheduler = adjust_lr(50, 0.1)



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
            {'params': net.module.parameters(), 'weight_decay': 1e-4, 'lr': args.lr, 'amsgrad': True},
            {'params':  criterion.parameters(), 'weight_decay': 1e-4, 'lr': args.lr}
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
        scheduler.adjust(optimizer, epoch)
        # epoch_start = time.time()
        for batch_id, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            hash_bits = net(imgs)
            loss_dual = criterion(hash_bits, labels)
            hash_binary = torch.sign(hash_bits)
            batchY = EncodingOnehot(labels, classes).cuda()
            W = torch.pinverse(batchY.t() @ batchY) @ batchY.t() @ hash_binary           # Update W

            batchB = torch.sign(torch.mm(batchY, W) + args.eta * hash_bits)  # Update B

            loss_vertex = (hash_bits - batchB).pow(2).sum() / len(imgs)     # quantization loss
            loss_h = loss_dual + args.eta * loss_vertex

            dcdh_loss.update(loss_h.item(), len(imgs))
            loss_h.backward()
            optimizer.step()

        print("[epoch: %d]\t[hashing loss: %.3f ]" % (epoch+1, dcdh_loss.avg))

        if (epoch+1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                # centers_trained = torch.sign(criterion.centers.data).cuda()
                trainB, train_labels = compute_result(trainloader, net, device)
                testB, test_labels = compute_result(testloader, net, device)
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
    %(args.dataset, args.len, args.bs, args.lr, EPOCHS))
    print("HyperParams:\nsigma: %.3f\t inner_param: %.4f\t eta: %.4f\t" % (args.sigma, args.inner_param, args.eta))
    train(EPOCHS)
