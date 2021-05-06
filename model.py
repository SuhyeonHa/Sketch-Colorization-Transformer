import torch
from torch import nn
import torch.nn.functional as F
import transformer
from torchvision.models import resnet50


class Sketch2Color(nn.Module):
    def __init__(self, args, pretrained=False):
        super(Sketch2Color, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, nc=3, LR=0.2):
                super(Encoder, self).__init__()
                self.layer1 = ConvBlock(nc, 16, LR=LR)  # 256
                self.layer2 = ConvBlock(16, 16, LR=LR)
                self.layer3 = ConvBlock(16, 32, stride=2, LR=LR)  # 128
                self.layer4 = ConvBlock(32, 32, LR=LR)
                self.layer5 = ConvBlock(32, 64, stride=2, LR=LR)  # 64
                self.layer6 = ConvBlock(64, 64, LR=LR)
                self.layer7 = ConvBlock(64, 128, stride=2, LR=LR)  # 32
                self.layer8 = ConvBlock(128, 128, LR=LR)
                self.layer9 = ConvBlock(128, 256, stride=2, LR=LR)  # 16
                self.layer10 = ConvBlock(256, 256, LR=LR)
                self.down_sampling = nn.AdaptiveAvgPool2d((16, 16))

            def forward(self, x):
                feature_map1 = self.layer1(x)
                feature_map2 = self.layer2(feature_map1)
                feature_map3 = self.layer3(feature_map2)
                feature_map4 = self.layer4(feature_map3)
                feature_map5 = self.layer5(feature_map4)
                feature_map6 = self.layer6(feature_map5)
                feature_map7 = self.layer7(feature_map6)
                feature_map8 = self.layer8(feature_map7)
                feature_map9 = self.layer9(feature_map8)
                feature_map10 = self.layer10(feature_map9)

                down_feature_map1 = self.down_sampling(feature_map1)
                down_feature_map2 = self.down_sampling(feature_map2)
                down_feature_map3 = self.down_sampling(feature_map3)
                down_feature_map4 = self.down_sampling(feature_map4)
                down_feature_map5 = self.down_sampling(feature_map5)
                down_feature_map6 = self.down_sampling(feature_map6)
                down_feature_map7 = self.down_sampling(feature_map7)
                down_feature_map8 = self.down_sampling(feature_map8)

                output = torch.cat([down_feature_map1,
                                    down_feature_map2,
                                    down_feature_map3,
                                    down_feature_map4,
                                    down_feature_map5,
                                    down_feature_map6,
                                    down_feature_map7,
                                    down_feature_map8,
                                    feature_map9,
                                    feature_map10,
                                    ], dim=1)

                feature_list = [feature_map1,
                                feature_map2,
                                feature_map3,
                                feature_map4,
                                feature_map5,
                                feature_map6,
                                feature_map7,
                                feature_map8,
                                feature_map9,
                                # feature_map10,
                                ]
                b, ch, h, w = output.size()
                return output, feature_list

        class Decoder(nn.Module):
            def __init__(self, spec_norm=False, LR=0.2):
                super(Decoder, self).__init__()
                self.layer10 = ConvBlock(992, 256, LR=LR)  # 16->16
                self.layer9 = ConvBlock(256+256, 256, LR=LR)  # 16->16
                self.layer8 = ConvBlock(256+128, 128, LR=LR, up=True)  # 16->32
                self.layer7 = ConvBlock(128+128, 128, LR=LR)  # 32->32
                self.layer6 = ConvBlock(128+64, 64, LR=LR, up=True)  # 32-> 64
                self.layer5 = ConvBlock(64+64, 64, LR=LR)  # 64 -> 64
                self.layer4 = ConvBlock(64+32, 32, LR=LR, up=True)  # 64 -> 128
                self.layer3 = ConvBlock(32+32, 32, LR=LR)  # 128 -> 128
                self.layer2 = ConvBlock(32+16, 16, LR=LR, up=True)  # 128 -> 256
                self.layer1 = ConvBlock(16+16, 16, LR=LR)  # 256 -> 256
                self.last_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

            def forward(self, x, feature_list):
                feature_map10 = self.layer10(x)
                feature_map9 = self.layer9(torch.cat([feature_map10, feature_list[-1]], dim=1))
                feature_map8 = self.layer8(feature_map9, feature_list[-2])
                feature_map7 = self.layer7(torch.cat([feature_map8, feature_list[-3]], dim=1))
                feature_map6 = self.layer6(feature_map7, feature_list[-4])
                feature_map5 = self.layer5(torch.cat([feature_map6, feature_list[-5]], dim=1))
                feature_map4 = self.layer4(feature_map5, feature_list[-6])
                feature_map3 = self.layer3(torch.cat([feature_map4, feature_list[-7]], dim=1))
                feature_map2 = self.layer2(feature_map3, feature_list[-8])
                feature_map1 = self.layer1(torch.cat([feature_map2, feature_list[-9]], dim=1))
                feature_map0 = self.last_conv(feature_map1)
                return feature_map0

        self.img_size = args.img_size
        self.ncluster = args.ncluster

        self.trans_dim = args.transformer_dim

        self.feature_size = self.img_size // 16
        self.n_patches = (self.img_size // 16) ** 2
        self.pos_embeddings_e = nn.Parameter(torch.zeros(self.n_patches, 1, self.trans_dim))
        # self.pos_embeddings_c = nn.Parameter(torch.zeros(1, self.n_patches, self.trans_dim))

        self.input_proj_e = nn.Conv2d(992, self.trans_dim, kernel_size=1)
        self.input_proj_c = nn.Conv2d(1792, self.trans_dim, kernel_size=1)
        self.proj_p = nn.Conv2d(self.ncluster*3, self.trans_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(self.trans_dim, 992, kernel_size=1)

        self.tf_proj = ConvBlock(self.trans_dim*2, self.trans_dim)

        self.gen_palette = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 64),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64, self.ncluster * 3),
        )

        self.upsample = nn.Upsample(scale_factor=16, mode='nearest')
        self.e_encoder = Encoder(nc=1)

        # encoder for color
        self.backbone = resnet50()
        del self.backbone.fc
        self.down_sampling = nn.AdaptiveAvgPool2d((16, 16))

        self.transformer = transformer.Transformer(d_model=self.trans_dim, dropout=args.dropout, nhead=args.nheads,
        dim_feedforward=args.transformer_dim, num_encoder_layers=args.num_enc_layers, num_decoder_layers=args.num_dec_layers,
        normalize_before=False, return_intermediate_dec=False)

        self.decoder = Decoder()


    def forward(self, edge, color): # torch.Size([B, 1, 256, 256]), torch.Size([B, 3, 256, 256])
        edge, e_feats = self.e_encoder(edge) # torch.Size([B, 992, 16, 16])

        x = self.backbone.conv1(color)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x) # torch.Size([8, 256, 64, 64])
        x2 = self.backbone.layer2(x1) # torch.Size([8, 512, 32, 32])
        x3 = self.backbone.layer3(x2) # torch.Size([8, 1024, 16, 16])
        # x4 = self.backbone.layer4(x3) # torch.Size([8, 2048, 8, 8])

        x1 = self.down_sampling(x1) # torch.Size([8, 256, 16, 16])
        x2 = self.down_sampling(x2) # torch.Size([8, 512, 16, 16])

        color = torch.cat((x1, x2, x3), dim=1) # torch.Size([8, 1792, 16, 16])

        edge = self.input_proj_e(edge) # torch.Size([2, 128, 16, 16])
        color = self.input_proj_c(color)
        palette = self.gen_palette(color) # torch.Size([B, 12, 1, 1])

        palette_tf = self.upsample(palette) # torch.Size([B, 12, 16, 16])
        palette_tf = self.proj_p(palette_tf) # torch.Size([B, 128, 16, 16])

        # tf_input = torch.cat((edge, palette_tf), dim=1)
        # tf_input = self.tf_proj(tf_input)
        # h = self.transformer(tf_input, self.pos_embeddings_e) # torch.Size([2, 256, 128])
        h = self.transformer(edge, palette_tf, self.pos_embeddings_e) # torch.Size([2, 256, 128])
        h = h.reshape(h.size(0),self.feature_size,self.feature_size,h.size(-1)) # torch.Size([2, 16, 16, 128])
        h = h.permute(0, 3, 1, 2) # torch.Size([2, 256, 16, 16])
        h = self.output_proj(h)  # torch.Size([2, 992, 16, 16])

        h = self.decoder(h, e_feats)
        return h, palette


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, LR=0.01, stride=1, up=False, pono=False, ms=False):
        super(ConvBlock, self).__init__()
        self.up = up
        if self.up:
            self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up_sample = None

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(LR, inplace=False)
        )

    def forward(self, x1, x2=None):
        if self.up_sample is not None:
            x1 = self.up_sample(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.main(x)
        else:
            return self.main(x1)