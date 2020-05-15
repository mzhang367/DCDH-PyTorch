from utils import *
from model import DFHNet
from Loss import DualClasswiseLoss
import torch.backends.cudnn as cudnn
import argparse
from torchvision import datasets
from torch.utils.data import DataLoader
import time

"""
Model configuration and evaluation option.
Accept multiple models with different bits length for the same dataset.
"""

parser = argparse.ArgumentParser(description='Evaluation on three datasets: {Facescrub, YouTubeFaces, VGGFace}; Supported metrics: {mAP, precision, recall, top-k}')
parser.add_argument('--load', type=str, help='Path to load the model')
parser.add_argument('--dataset', type=str, default= 'facescrub', help='should be one of {facescrub, youtube, vgg}')
parser.add_argument('-o', '--option', action='store_true', help='specify which metric to evaluate')
parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
parser.add_argument('--len', type=int, default=48,
                    help='length of hashing codes,  should be one of {12, 24, 36, 48}')

args = parser.parse_args()

model_path = args.load
bits = args.len
dataset = args.dataset
option = args.option
device = "cuda:0" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True



def evaluation(model_path, bits, dataset, option):

    if dataset in ['facescrub', 'youtube']:

        transform_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = MyDataset(dataset, transform=transform_tensor, train=True)
        trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)
        classes = len(np.unique(trainset.train_y))
        testset = MyDataset(dataset, transform=transform_tensor, train=False)
        testloader = DataLoader(testset, batch_size=args.bs, shuffle=False)
    else:
        trainPaths = "./vgg_face2/train"
        testPaths = "./vgg_face2/test"
        Normalize = transforms.Normalize((0.5141, 0.4074, 0.3588), (1, 1, 1))
        transform_train = transforms.Compose([
        transforms.Resize(160),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Normalize,
        ])

        transform_validation = transforms.Compose([
            transforms.Resize(160),
            transforms.ToTensor(),
            Normalize,
        ])
        trainset = datasets.ImageFolder(root=trainPaths, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=8)

        testset = datasets.ImageFolder(root=testPaths, transform=transform_validation)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=8)

    top_list = torch.tensor([1, 5, 10]).int().tolist()

    if not option:

        checkpoint = torch.load('./checkpoint/%s' % model_path)
        print("evaluation on %s" % model_path)
        net = torch.nn.DataParallel(DFHNet(bits)).to(device)
        criterion = DualClasswiseLoss(num_classes=classes, inner_param=0.1, sigma=0.25, feat_dim=bits, use_gpu=True)
        net.load_state_dict(checkpoint['backbone_state_dict'])
        criterion.load_state_dict(checkpoint['clf_state_dict'])
        centers_trained = torch.sign(criterion.centers.data)
        net.eval()
        with torch.no_grad():
            trainB, train_labels = compute_result(trainloader, net, device, centers_trained)
            testB, test_labels = compute_result(testloader, net, device, centers_trained)
            since = time.time()
            mAP = compute_mAP(trainB, testB, train_labels, test_labels, device)
            time_elapsed = time.time() - since
            print('[Evaluate Phase] MAP: %.2f%%' % (100. * float(mAP)))
            print("Calculate mAP in {:.0f} min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            print("=================================")

            top_results = compute_topK(trainB, testB, train_labels, test_labels, device, top_list)
            for i in range(len(top_list)):
                print("top%d: %.2f%%" %(top_list[i], top_results[i]))
            print("=================================")

            precision, recall = evaluate_recall_pre(train_labels, test_labels, trainB, testB, device)
            print('Precision with Hamming radius_2 : {:.2%}'.format(precision))
            print('Recall with Hamming radius_2 : {:.2%}'.format(recall))
