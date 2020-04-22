import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
FACE_FEATURES = 1024


class DCFH_BN2(nn.Module):
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
        super(DCFH_BN2, self).__init__()

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
            nn.Conv2d(64, 128, 2, 2),   # kernel =2, stride = 2
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True))# size = 4

        self.upscales_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),# size = 8
            nn.ReLU(True)
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.upscales_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),# size = 16
            nn.ReLU(True)
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.upscales_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),#size = 32, out_channel = 16
            nn.ReLU(True)
        )
        self.bn3 = nn.BatchNorm2d(16)

        self.upscales_4 = nn.Sequential(#out_channel = 1, w = 1*32*32
            nn.Conv2d(16, 3, 1)# 16-->3
        )

        ############################################################################strategy 1
        self.attention_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.MaxPool2d
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

        #self.mask_attention = nn.Sigmoid()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),#30
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),#15

            nn.Conv2d(32, 64, 2),#14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),#7

            nn.Conv2d(64, 128, 2),#6
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2) #output_size: 3*3
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2),#output_size: 2*2
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.channel_attention3 = nn.Sequential(
            nn.Linear(128*3*3, 512),
            nn.ReLU(True),
            nn.Linear(512, 128) # apply on channel!
        )

        self.channel_attention4 = nn.Sequential(# Note: apply on channel!
            nn.Linear(256*2*2, 512),  # follow conv4
            nn.ReLU(True),
            nn.Linear(512, 256)
        )

        self.face_features_layer = nn.Sequential(
            nn.Linear(2176, FACE_FEATURES),
            nn.BatchNorm1d(FACE_FEATURES),
            nn.ReLU(True)
        )

        self.hash_layer = nn.Sequential(
            nn.Linear(FACE_FEATURES, hash_bits),
            nn.BatchNorm1d(hash_bits),
            #nn.Tanh() ######################################### comment or not
        )

        '''self.classifier = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hash_bits, CLASS_NU),    # (n, class_num)
            nn.LogSoftmax(dim=1) #  log(softmax(x)) function
        )'''
        self.apply(self.weights_init)

    def forward(self, x):

        attention_mask_16 = self.spatial_features_1(x)
        attention_mask_8 = self.spatial_features_2(attention_mask_16)    #FCN
        attention_mask_4 = self.spatial_features_3(attention_mask_8)
        attention_mask = self.fc(attention_mask_4)
        attention_mask = self.upscales_1(attention_mask)
        attention_mask = self.bn1(attention_mask + attention_mask_8)# up_scale + downscale (same size 8)
        attention_mask = self.upscales_2(attention_mask)
        attention_mask = self.bn2(attention_mask + attention_mask_16)#up_scale + downscale (same size 16)
        attention_mask = self.upscales_3(attention_mask)
        attention_mask = self.upscales_4(attention_mask)#size: len(x)*3*H*W
        attention_mask = torch.sigmoid(attention_mask)## here we change to spatial attention, or sth else #######################



        # spatial normalization
        '''spatial_attention_min, _ = torch.min(attention_mask.view(x.size(0), -1), dim=1)#(len(x), )
        spatial_attention_max, _ = torch.max(attention_mask.view(x.size(0), -1), dim=1)

        spatial_attention_min = spatial_attention_min.reshape(x.size(0), 1, 1, 1)
        spatial_attention_max = spatial_attention_max.reshape(x.size(0), 1, 1, 1)
        attention_mask = (attention_mask - spatial_attention_min)/(spatial_attention_max - spatial_attention_min)'''

        ############ start my preformance
        feature_trunk = self.attention_conv1(x)
        feature_trunk = self.attention_conv2(feature_trunk)
        feature_trunk = self.attention_conv3(feature_trunk)
        feature_trunk = self.attention_conv4(feature_trunk)


        # branch2_weight* branch2
        x_with_mix_attention = attention_mask * feature_trunk
        feature_catenate = feature_trunk + x_with_mix_attention
        #x_with_mix_attention += feature_trunk


        features_3 = self.features(feature_catenate)
        features_4 = self.conv4(features_3)

        '''channel_attention_3 = self.channel_attention3(features_3.view(features_3.size(0), -1))   # n*(128*3*3) --> n*128
        channel_attention_4 = self.channel_attention4(features_4.view(features_4.size(0), -1))   # n*(256*2*2) --> n*256

        # channel normalization
        channel_attention_3_min, _ = torch.min(channel_attention_3, dim=1)
        channel_attention_3_max, _ = torch.max(channel_attention_3, dim=1)
        channel_attention_3_min = channel_attention_3_min.unsqueeze(1)
        channel_attention_3_max = channel_attention_3_max.unsqueeze(1)
        #(n*512 - n_min)/ (n_max - n_min)
        channel_attention_3 = (channel_attention_3 - channel_attention_3_min)/(channel_attention_3_max - channel_attention_3_min)
        channel_attention_4_min, _ = torch.min(channel_attention_4, dim=1)
        channel_attention_4_max, _ = torch.max(channel_attention_4, dim=1)
        channel_attention_4_min = channel_attention_4_min.unsqueeze(1)
        channel_attention_4_max = channel_attention_4_max.unsqueeze(1)
        channel_attention_4 = (channel_attention_4 - channel_attention_4_min)/(channel_attention_4_max - channel_attention_4_min)
        features_3_with_attention = features_3 * channel_attention_3.reshape(channel_attention_3.size(0), -1, 1, 1)  #(N*128*3*3) * (N*128*1*1)
        features_4_with_attention = features_4 * channel_attention_4.reshape(channel_attention_4.size(0), -1, 1, 1)  #(N*256*2*2) * (N*256*1*1)'''

        #fuse features: 128*3*3+256*2*2 = 2176
        features_a = torch.cat([features_3.view(features_3.size(0), -1), features_4.view(features_4.size(0), -1)], -1)
        features_a = self.face_features_layer(features_a) # 2176--> 1024
        hash_a = self.hash_layer(features_a)

        return hash_a


class DCFH_advance(nn.Module):

    '''No extra modules or defined modules without use'''

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
        super(DCFH_advance, self).__init__()

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
            nn.Conv2d(64, 128, 2, 2),   # kernel =2, stride = 2
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True))# size = 4

        self.upscales_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),# size = 8
            nn.ReLU(True)
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.upscales_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),# size = 16
            nn.ReLU(True)
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.upscales_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),#size = 32, out_channel = 16
            nn.ReLU(True)
        )
        self.bn3 = nn.BatchNorm2d(16)

        self.upscales_4 = nn.Sequential(#out_channel = 1, w = 1*32*32
            nn.Conv2d(16, 3, 1)# 16-->3
        )

        ############################################################################strategy 1
        self.attention_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.MaxPool2d
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

        #self.mask_attention = nn.Sigmoid()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),    # 30
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),     # 15

            nn.Conv2d(32, 64, 2),   # 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),     # 7

            nn.Conv2d(64, 128, 2),  # 6
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)      # output_size: 3*3
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2),     # output_size: 2*2
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
            #nn.Tanh() ######################################### comment or not
        )

        self.apply(self.weights_init)

    def forward(self, x):

        attention_mask_16 = self.spatial_features_1(x)
        attention_mask_8 = self.spatial_features_2(attention_mask_16)    #FCN
        attention_mask_4 = self.spatial_features_3(attention_mask_8)
        attention_mask = self.fc(attention_mask_4)
        attention_mask = self.upscales_1(attention_mask)
        attention_mask = self.bn1(attention_mask + attention_mask_8)# up_scale + downscale (same size 8)
        attention_mask = self.upscales_2(attention_mask)
        attention_mask = self.bn2(attention_mask + attention_mask_16)#up_scale + downscale (same size 16)
        attention_mask = self.upscales_3(attention_mask)
        attention_mask = self.upscales_4(attention_mask)#size: len(x)*3*H*W
        attention_mask = torch.sigmoid(attention_mask)## here we change to spatial attention, or sth else #######################

        ############ start my preformance
        feature_trunk = self.attention_conv1(x)
        feature_trunk = self.attention_conv2(feature_trunk)
        feature_trunk = self.attention_conv3(feature_trunk)
        feature_trunk = self.attention_conv4(feature_trunk)


        # branch2_weight* branch2
        x_with_mix_attention = attention_mask * feature_trunk
        feature_catenate = feature_trunk + x_with_mix_attention
        #x_with_mix_attention += feature_trunk


        features_3 = self.features(feature_catenate)
        features_4 = self.conv4(features_3)

        features_a = torch.cat([features_3.view(features_3.size(0), -1), features_4.view(features_4.size(0), -1)], -1)
        features_a = self.face_features_layer(features_a) # 2176--> 1024
        hash_a = self.hash_layer(features_a)

        return hash_a


if __name__ == '__main__':

    fake_data = torch.randn(2, 3, 32, 32)
    net = DCFH_BN2(48)
    #print(net)
    net(fake_data)
