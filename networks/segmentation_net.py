import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=16):
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = self._encoder_block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.encoder2 = self._encoder_block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.encoder3 = self._encoder_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        self.bottleneck = self._decoder_block(features * 4, features * 8)

        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=3, stride=(2, 2, 1), padding=(1,1,1), output_padding=[1,1,0]
        )
        self.decoder3 = self._decoder_block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=3, stride=(2, 2, 1), padding=(1,1,1), output_padding=[1,1,0]
        )
        self.decoder2 = self._decoder_block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=3, stride=(2, 2, 1), padding=(1,1,1), output_padding=[1,1,0]
        )
        self.decoder1 = self._decoder_block(features * 2, features)

        self.conv = nn.ConvTranspose3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )


    def forward(self, x):
        enc1 = self.encoder1(x)
        # print(enc1.size())

        enc2 = self.encoder2(self.pool1(enc1))
        # print(enc2.size())

        enc3 = self.encoder3(self.pool2(enc2))
        # print(enc3.size())

        bottleneck = self.bottleneck(self.pool3(enc3))
        # print(bottleneck.size())

        dec3 = self.upconv3(bottleneck)
        # print(dec3.size())
        # print(enc3.size())

        dec3 = torch.cat((dec3, enc3), dim=1)
        # print(dec3.size())

        dec3 = self.decoder3(dec3)
        # print(dec3.size())

        dec2 = self.upconv2(dec3)
        # print(dec2.size())
        dec2 = torch.cat((dec2, enc2), dim=1)
        # print(dec2.size())
        dec2 = self.decoder2(dec2)
        # print(dec2.size())

        dec1 = self.upconv1(dec2)
        # print(dec1.size())
        dec1 = torch.cat((dec1, enc1), dim=1)
        # print(dec1.size())
        dec1 = self.decoder1(dec1)


        return self.conv(dec1)

    def _encoder_block(self, in_channels, features):
        layers = [
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.Conv3d(features, features, kernel_size=3,  padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channels, features):
        layers = [
            nn.Conv3d(in_channels, features, kernel_size=3,  padding=1, bias=False),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)
