# coding: utf-8
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils import *
#from model import *
#from model_attention import DCFH_a
#model_testX (DCFH_BN) + DCFH_fast_semantics
#model_attent(DCFH_a) + train_classwise, former best
#model_test(DCFH_dir) + train_classwise
import pdb
from model_attention2 import DCFH_BN2, DCFH_advance
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
import sys
#from VisTool import VisdomLinePlotter
import argparse
from BeaconLoss_Gaussian import DualclasswiseLoss
import torch.backends.cudnn as cudnn
#########test lr from 0.01 reduce

parser = argparse.ArgumentParser(description='PyTorch Deep Attention-aware Face Hashing Imple.')
#parser.add_argument('--lr1', default=0.001, type=float, help='learning rate spatial layer')
parser.add_argument('--lr1', default=0.005, type=float, help='learning rate hashing layer')
parser.add_argument('--lr2', default=0.005, type=float, help='learning rate hashing layer')


#parser.add_argument('--freq', default=300, type=int, help='freq. of print batch information')
parser.add_argument('-r', '--up_c', action='store_true', help='update centroids manually instead of gradient descent')

parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
parser.add_argument('--save', type=str, help='path to saving model')
# parser.add_argument('--sch', type=str, default='plateau', help='learning rate schedule')
parser.add_argument('--load', type=str, help='model path to evaluate')
parser.add_argument('--dataset', type=str, default= 'facescrub', help='which dataset for training.(facescrub, youtube)')
# parser.add_argument('-t', '--test', action='store_true', help='test mode turned on')
parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
#parser.add_argument('--network', type=str, default='google',
                    #help='Which network for train. (google, res50)')
parser.add_argument('--len', type=int, default=48,
                    help='number of bit for hashing. (16, 32, 48, 64)')

args = parser.parse_args()


EPOCHS = 700
transform_tensor = transforms.Compose([
    transforms.ToTensor()
])
trainset = MyDataset(args.dataset, transform=transform_tensor, train=True)
train_loader = DataLoader(trainset, batch_size=args.bs, shuffle=True)
classes = len(np.unique(trainset.train_y))
testset = MyDataset(args.dataset, transform=transform_tensor, train=False)
test_loader = DataLoader(testset, batch_size=args.bs, shuffle=False)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("using: " + str(device))
bits = args.len
#CUDA_LAUNCH_BLOCKING=1
#net = torch.nn.DataParallel(DCFH_advance(bits)).to(device)
net = DCFH_advance(bits).to(device)

cudnn.benchmark = True
criterion = DualclasswiseLoss(num_classes=classes, inner_param=0.1, sigma=0.25, feat_dim=args.len, update_centroids=args.up_c, use_gpu=True)


def GenerateCode(model, data_loader, num_data, bit):
    B = np.zeros((num_data, bit), dtype=np.float32)
    labels = np.ndarray(num_data, dtype=np.int)
    for iter, (data, target) in enumerate(data_loader):
        data_input, target = data.cuda(), target
        output = model(data_input)
        B[iter * args.bs: iter * args.bs + len(data), :] = torch.sign(output).detach().cpu().numpy() #tanh used after last fully connected layer
        labels[iter * args.bs: iter * args.bs + len(data)] = target.numpy().flatten()

    return B, labels


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = []
    lr.append(args.lr1 * (0.5 ** (epoch // 100)))
    lr.append(args.lr2 * (0.5 ** (epoch // 100)))
    #lr = args.lr * (0.5 ** (epoch // 100))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr[i]
    return lr


def EncodingOnehot(target, nclasses):
    target_onehot = torch.Tensor(target.size(0), nclasses).cuda()
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)#
    return target_onehot


def centers_computing(model, data_loader, num_data):
    U = np.zeros((num_data, args.len), dtype=np.float32)#output of hashing layer
    labels = np.ndarray(num_data, dtype=np.int)
    centers = np.ndarray((len(np.unique(trainset.train_y)), args.len))
    for iter, (data, target) in enumerate(data_loader):
        data_input, target = data.cuda(), target
        output = model(data_input)
        U[iter * args.bs: iter * args.bs + len(data), :] = output.detach().cpu().numpy()
        labels[iter * args.bs: iter * args.bs + len(data)] = target.numpy().flatten()
    #print(np.unique(labels))
    for i in np.unique(labels):
        index_list = np.where(labels==i)[0]
        centers[i, :] = U[index_list, :].sum(axis=0)/len(index_list)
    return centers

if not args.evaluate:
    ###################################################################################################

    print('==> Preparing training data..')

    print("number of classes: ", len(np.unique(trainset.train_y)))
    print("number of training images: ", len(trainset.train_y))
    print("number of training batches per epoch:", len(train_loader))
    print("number of test images: ", len(testset.test_y))
    print("number of testing batches per epoch:", len(test_loader))
    # pdb.set_trace()

    ##################################################################################################
    centers = torch.randn(classes, args.len).cuda().detach()
    centers_recordings = np.ndarray((EPOCHS, classes, args.len))
    best_MAP = 0
    best_epoch = 1  # best test accuracy
    best_loss = 1e6
    # start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    print('==> Building model..')

    optimizer = optim.Adam([
        {'params': net.module.parameters(), 'weight_decay': 1e-4, 'lr': args.lr1, 'amsgrad': True}, # module
        {'params':  criterion.parameters(), 'weight_decay': 1e-4, 'lr': args.lr2}
    ])
    # alpha = 0.01
    lambda_miu = 1
    eta = 0.01################################### 0.005 0.001
    eta_miu = 0.01
    #weight_cubic = 10
    #weight_vertex = 0.01


def train(epoch):
    print('==> Epoch: %d' % (epoch + 1))

    net.train()
    global centers
    global best_MAP
    global best_epoch
    global best_loss
    global centers_recordings
    dcdh_loss = AverageMeter()
    adjust_learning_rate(optimizer, epoch)
    #schedule.step(epoch)

    for batch_id, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        hash_bits = net(imgs)
        loss_dual = criterion(hash_bits, labels)#be careful require on dim.
        hash_binary = torch.sign(hash_bits)
        batchY = EncodingOnehot(labels, classes).cuda() # torch.tensor
        #batchY.to(device)
        W = torch.mm(torch.inverse(torch.mm(torch.transpose(batchY, 0, 1), batchY) + lambda_miu*torch.eye(batchY.size(1)).cuda()),
        torch.mm(torch.transpose(batchY, 0, 1), hash_binary)).detach()              # Update W

        batchB = torch.sign(torch.mm(batchY, W) + eta_miu*hash_bits).detach()       # Update B

        loss_vertex = (hash_bits - batchB).pow(2).sum() / len(imgs)
        loss_h = loss_dual + eta * loss_vertex

        dcdh_loss.update(loss_h.item(), len(imgs))
        loss_h.backward(retain_graph=True)
        optimizer.step()

    centers_recordings[epoch, :, :] = criterion.centers.data.cpu().numpy()

    #plotter.plot(dcdh_loss.avg, epoch + 1, 'Loss in Training', 'hashing loss', 'loss', 'epoch')

    print("[epoch: %d]\t[hashing loss: %.3f ]" % (epoch+1, dcdh_loss.avg))
    if args.up_c:
        if (epoch+1) % 2 == 0:
            net.eval()
            # valid_loss = AverageMeter()
            with torch.no_grad():
                centers = torch.from_numpy(centers_computing(net, train_loader, len(trainset))).float().cuda().detach()

    if (epoch+1) % 10 == 0:# 10 epoch
        net.eval()
        with torch.no_grad():

            trainB, train_labels = compute_result(train_loader, net, device)
            testB, test_labels = compute_result(test_loader, net, device)
            mAP, _ = compute_mAP(trainB, testB, train_labels, test_labels, device)
            if mAP > best_MAP:
                best_MAP = mAP

        print("[epoch: %d]\t[hashing loss: %.3f\t [mAP: %.2f%%]" % (epoch+1, dcdh_loss.avg, float(mAP)*100.0))

    if dcdh_loss.avg < best_loss:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save({'backbone_state_dict': net.state_dict(),
                    'clf_state_dict': criterion.state_dict()}, './checkpoint/%s' % args.save)
        best_loss = dcdh_loss.avg
        best_epoch = epoch + 1

    return dcdh_loss.avg
    ##############################################
    # adjust_learning_rate(optimizer, epoch, args)##


def test():
    assert os.path.exists(os.path.join("./checkpoint", args.load)), "model path not found!"
    checkpoint = torch.load("./checkpoint/%s" %args.load)
    net.load_state_dict(checkpoint['backbone_state_dict'])
    criterion.load_state_dict(checkpoint['clf_state_dict'])

    '''model = torch.nn.DataParallel(net)

    torch.save({'backbone_state_dict': model.state_dict(),
                    'clf_state_dict': criterion.state_dict()}, './checkpoint/%s' % args.save)'''

    #pdb.set_trace()
    #net.load_state_dict(checkpoint)
    net.eval()
    with torch.no_grad():

        since = time.time()
        trainB, train_labels = compute_result(train_loader, net, device)
        testB, test_labels = compute_result(test_loader, net, device)
        MAP, top_k = compute_mAP(trainB, testB, train_labels, test_labels, device, top=50)
        precision, recall = evaluate_recall_pre(train_labels, test_labels, trainB, testB, device)
        time_elapsed = time.time() - since
        print("Calculate mAP in {:.0f} min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(MAP), 100. * float(top_k)))
        print('Precision with Hamming radius_2 : {:.2%}'.format(precision))
        print('Recall with Hamming radius_2 : {:.2%}'.format(recall))





if __name__ == '__main__':


    if not os.path.isdir('log'):
        os.mkdir('log')
    save_dir = './log'

    if args.evaluate:
        test()

    else:
        assert args.save
        # log_trainloss = []
        # log_testloss = []
        sys.stdout = Logger(os.path.join(save_dir,
        str(args.len) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
        print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n Up_center: %r\n Sigma: %.2f\n #Epoch: %d\n Inner_param: %.3f"
              %(args.dataset, args.len, args.bs, args.up_c, criterion.sigma, EPOCHS, criterion.inner_param))

        print("eta: %.4f, lambda_miu: %.4f, eta_miu: %.4f" %(eta, lambda_miu, eta_miu))

        #plotter = VisdomLinePlotter(env_name='main')

        since = time.time()
        # train start!
        for epoch in range(EPOCHS):

            train_loss = train(epoch)
            #plotter.plot(train_loss, epoch + 1, 'loss in Training', 'hashing loss', 'loss', 'epoch')
            # [~, ~, Title, legend, ylabel, xlabel]
            # vis.plot_curves({'train_acc': train_acc, 'val_acc': acc}, iters=epoch, title='Accuracy', xlabel='Epoch', ylabel='acc(%)')
            '''if (epoch + 1 - best_epoch) > 120:
                print("Training terminated at epoch %d" %(epoch + 1))
                break'''

        time_elapsed = time.time() - since
        print("Training Completed in {:.0f}min {:.0f}s with best mAP {:.2%}".format(time_elapsed // 60, time_elapsed % 60, float(best_MAP)))
        print("Tracking center points saved...")
        np.save('center' + '_' + datetime.now().strftime('%m%d%H%M') + '.npy', centers_recordings)
        print("Model saved as %s" % args.save)
