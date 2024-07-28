import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, input_channels, filters, kernel_size=(3, 3)):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding='same')
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        pool = self.pool(x)
        return x, pool
    

class Encoder(nn.Module):
    def __init__(self, filters=(64, 128, 256, 512, 1024)):
        super(Encoder, self).__init__()
        self.enc1 = Block(3, filters[0])
        self.enc2 = Block(filters[0], filters[1])
        self.enc3 = Block(filters[1], filters[2])
        self.enc4 = Block(filters[2], filters[3])
        self.enc5 = Block(filters[3], filters[4])

    def forward(self, x):
        x1, p1 = self.enc1(x)
        x2, p2 = self.enc2(p1)
        x3, p3 = self.enc3(p2)
        x4, p4 = self.enc4(p3)
        x5, p5 = self.enc5(p4)

        return x1, x2, x3, x4, x5, p5
    

class Decoder(nn.Module):
    def __init__(self, filters=(1024, 512, 256, 128, 64)):
        super(Decoder, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(filters[0] + filters[1], filters[1], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], filters[1], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(filters[1] + filters[2], filters[2], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            nn.Conv2d(filters[2] + filters[3], filters[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[3], filters[3], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = nn.Sequential(
            nn.Conv2d(filters[3] + filters[4], filters[4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[4], filters[4], 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoder_outputs):
        x1, x2, x3, x4, x5, p5 = encoder_outputs

        x = self.up1(p5)
        x = self.adjust_size(x, x4)
        x = torch.cat([x, x4], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = self.adjust_size(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = self.adjust_size(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x = self.adjust_size(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.dec4(x)

        return x

    def adjust_size(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = F.interpolate(x, size=target.size()[2:], mode='bilinear', align_corners=True)
        return x
    

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        x = self.decoder(encoder_outputs)
        outputs = self.final_conv(x)
        return outputs