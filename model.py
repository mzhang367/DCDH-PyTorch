import torch
import torch.nn as nn
import torch.nn.init as init
FACE_FEATURES = 1024


class DFHNet(nn.Module):

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


if __name__ == '__main__':

    fake_data = torch.randn(2, 3, 32, 32)
    net = DFHNet(48)
    net(fake_data)
