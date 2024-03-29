from utils import *
from model import DFHNet, SphereNet_hashing
import torch.backends.cudnn as cudnn
import argparse
from torchvision import datasets
from torch.utils.data import DataLoader
import time

"""
Model configuration and evaluation.
Support evaluation metrics: mAP, top-k, Precision@H=2, Recall@H=2
"""

parser = argparse.ArgumentParser(description='Evaluation on three datasets: {Facescrub, YouTubeFaces, VGGFace}; Supported metrics: {mAP, precision, recall, top-k}')
parser.add_argument('--load', type=str, help='Path to load the model')
parser.add_argument('--dataset', type=str, default='facescrub', help='should be one of {facescrub, youtube, vgg}')
parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
parser.add_argument('--len', type=int, default=48,
                    help='length of hashing codes,  should be one of {12, 24, 36, 48}')

args = parser.parse_args()

model_path = args.load
bits = args.len
dataset = args.dataset
device = "cuda:0" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True


def evaluation(model_path, bits, dataset):

    if dataset in ['facescrub', 'youtube']:

        transform_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = MyDataset(dataset, transform=transform_tensor, train=True)
        trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)
        classes = len(np.unique(trainset.train_y))
        testset = MyDataset(dataset, transform=transform_tensor, train=False)
        testloader = DataLoader(testset, batch_size=args.bs, shuffle=False)
        net = DFHNet(bits)

    else:
        trainPaths = "./vgg_face2/train"
        testPaths = "./vgg_face2/test"
        Normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform_test = transforms.Compose([
                    transforms.Resize(120),
                    transforms.CenterCrop(112),
                    transforms.ToTensor(),
                    Normalize])

        trainset = datasets.ImageFolder(root=trainPaths, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)

        testset = datasets.ImageFolder(root=testPaths, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)
        net = SphereNet_hashing(num_layers=20, hashing_bits=bits)
    net.to(device)
    top_list = torch.tensor([1, 5, 10]).int().tolist()

    checkpoint = torch.load('./checkpoint/%s' % model_path)
    print("evaluation on %s" % model_path)
    net.load_state_dict(checkpoint['backbone'])

    net.eval()
    with torch.no_grad():
        trainB, train_labels = compute_result(trainloader, net, device)
        testB, test_labels = compute_result(testloader, net, device)
        since = time.time()
        mAP = compute_mAP(trainB, testB, train_labels, test_labels, device)
        time_elapsed = time.time() - since
        print('[Evaluate Phase] MAP: %.2f%%' % (100. * float(mAP)))
        print("Calculate mAP in {:.0f} min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("=================================")

        top_results = compute_topK(trainB, testB, train_labels, test_labels, device, top_list)
        for i in range(len(top_list)):
            print("top%d: %.2f%%" %(top_list[i], 100. * top_results[i]))
        print("=================================")

        precision, recall = evaluate_recall_pre(train_labels, test_labels, trainB, testB, device)
        print('Precision with Hamming radius_2 : {:.2%}'.format(precision))
        print('Recall with Hamming radius_2 : {:.2%}'.format(recall))


if __name__ == '__main__':

    assert os.path.exists(os.path.join("./checkpoint", args.load)),  "invalid model path in the default directory"
    evaluation(model_path, bits, dataset)
