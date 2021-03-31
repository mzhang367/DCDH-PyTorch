import torch
import torch.nn as nn
import torch.nn.init as init
FACE_FEATURES = 1024


class DFHNet(nn.Module):
    """
    our designed backbone in the paper "Deep Center-Based Dual-Constrained Hashing for Discriminative Face Image Retrieval (DCDH)"
    """
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('ConvTranspose2d') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)

    def __init__(self, hash_bits):
        super(DFHNet, self).__init__()

        self.spatial_features_1 = nn.Sequential(
            nn.Conv2d(3, 32, 2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.spatial_features_2 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.spatial_features_3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True))    # size of output: 4

        self.upscales_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(True)
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.upscales_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),    # size of output: 16
            nn.ReLU(True)
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.upscales_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),#size = 32, out_channel = 16
            nn.ReLU(True)
        )
        self.bn3 = nn.BatchNorm2d(16)

        self.upscales_4 = nn.Sequential(
            nn.Conv2d(16, 3, 1)
        )

        self.attention_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.attention_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.attention_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.attention_conv4 = nn.Sequential(
            nn.Conv2d(128, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.face_features_layer = nn.Sequential(
            nn.Linear(2176, FACE_FEATURES),
            nn.BatchNorm1d(FACE_FEATURES),
            nn.ReLU(True)
        )

        self.hash_layer = nn.Sequential(
            nn.Linear(FACE_FEATURES, hash_bits),
            nn.BatchNorm1d(hash_bits),
        )

        self.apply(self.weights_init)

    def forward(self, x):
        """
        args:
            x: input, size of 32 * 32
        """
        # residual branch, encoder
        attention_mask_16 = self.spatial_features_1(x)
        attention_mask_8 = self.spatial_features_2(attention_mask_16)
        attention_mask_4 = self.spatial_features_3(attention_mask_8)
        attention_mask = self.fc(attention_mask_4)    # 4 * 4

        # residual branch, decoder
        attention_mask = self.upscales_1(attention_mask)
        attention_mask = self.bn1(attention_mask + attention_mask_8)
        attention_mask = self.upscales_2(attention_mask)
        attention_mask = self.bn2(attention_mask + attention_mask_16)
        attention_mask = self.upscales_3(attention_mask)
        attention_mask = self.upscales_4(attention_mask)
        attention_mask = torch.sigmoid(attention_mask)   # 32 * 32

        # trunk branch
        feature_trunk = self.attention_conv1(x)
        feature_trunk = self.attention_conv2(feature_trunk)
        feature_trunk = self.attention_conv3(feature_trunk)
        feature_trunk = self.attention_conv4(feature_trunk)    # 32 * 32

        # element-wise product and sum
        x_with_mix_attention = attention_mask * feature_trunk
        feature_catenate = feature_trunk + x_with_mix_attention    # 32 * 32

        # conv. block
        features_3 = self.features(feature_catenate)    # size of output: 3 * 3 * 128
        features_4 = self.conv4(features_3)    # size of output: 2 * 2 * 256

        features_a = torch.cat([features_3.view(features_3.size(0), -1), features_4.view(features_4.size(0), -1)], -1)    # fusion layer
        features_a = self.face_features_layer(features_a)    # fc layer, 2176--> 1024
        hash_a = self.hash_layer(features_a)    # hashing layer, 1024 --> num_bits

        return hash_a


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu1 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu2 = nn.PReLU(channels)
    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)

        return x + short_cut


class SphereNet_hashing(nn.Module):
    """
    The networks used in paper: "SphereFace: Deep Hypersphere Embedding for Face Recognition"
    """
    def __init__(self, num_layers=64, hashing_bits=48, clf=None):
        super().__init__()
        assert num_layers in [20, 64], 'spherenet num_layers should be 20 or 64'
        if num_layers == 20:
            layers = [1, 2, 4, 1]
        elif num_layers == 64:
            layers = [3, 8, 16, 3]
        else:
            raise ValueError('sphere' + str(num_layers) + "is not supported!")
        filter_list = [3, 64, 128, 256, 512]
        block = Block
        self.clf = clf
        self.hashing_bits = hashing_bits
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        self.fc = nn.Linear(512*7*7, 512)
        self.bn = nn.BatchNorm1d(512)
        self.logits = nn.Linear(512, self.hashing_bits)
        self.bn_last = nn.BatchNorm1d(self.hashing_bits)
        self.drop = nn.Dropout()

    def _make_layer(self, block, inplanes, planes, num_units, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.PReLU(planes))
        for i in range(num_units):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        x = self.bn(x)
        # x = self.drop(x)
        x = self.logits(x)
        out = self.bn_last(x)

        return out



if __name__ == '__main__':

    fake_data = torch.randn(2, 3, 32, 32)
    net = DFHNet(48)
    net(fake_data)
