from . import *
from .blocks.DWTNet import DWTBlock
from .blocks.UNet import UNet
from .blocks.dwt2d import DWTForward


class Encoder_MP(nn.Module):
    '''
	Insert a watermark into an image
	'''

    def __init__(self, H, W, message_length, blocks=4, channels=32):
        super(Encoder_MP, self).__init__()
        self.H = H
        self.W = W
        self.channels = channels
        message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))

        self.dwt_forward = DWTForward(J=1, wave='db1', mode='zero')
        self.image_pre_layer = ConvBNRelu(3, channels)
        self.image_first_layer = SENet(channels, channels, blocks=blocks)

        self.message_pre_layer = nn.Sequential(
            ConvBNRelu(1, channels),
            ExpandNet(channels, channels, blocks=message_convT_blocks),
        )

        self.after_concat_layer = nn.Sequential(
            UNet(dim=2 * channels),
            ConvBNRelu(2 * channels, channels),
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(channels + 3, 3, kernel_size=1),
        )

        self.message_conv1 = nn.Sequential(
            ConvBNRelu(channels, channels),
        )
        self.message_conv2 = nn.Sequential(
            ConvBNRelu(channels, channels),
        )
        self.image_conv1 = nn.Sequential(
            ConvBNRelu(3, channels),
        )
        self.image_dwt2 = nn.Sequential(
            DWTBlock(3, channels),
        )
        self.normrelu1 = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.normrelu2 = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.embeding = UNet(dim=2 * channels)
        self.cat_conv = nn.Sequential(

            ConvBNRelu(2 * channels, channels),
        )

    def forward(self, image, message):
        # Message Processor
        size = int(np.sqrt(message.shape[1]))

        message_image = message.view(-1, 1, size, size)
        message_pre = self.message_pre_layer(message_image)
        intermediate2 = message_pre

        message_pre_conv1 = self.message_conv1(intermediate2)
        message_pre_conv2 = self.message_conv2(intermediate2)

        image_pre_conv1 = self.image_conv1(image)
        image_pre_dwt2, Yl, LH, HL, HH = self.image_dwt2(image)

        fmi_conv_conv = message_pre_conv1 + image_pre_conv1
        fmi_conv_conv = self.normrelu1(fmi_conv_conv)
        fmi_conv_dwt = message_pre_conv2 + image_pre_dwt2
        fmi_conv_dwt = self.normrelu2(fmi_conv_dwt)

        output1 = torch.cat([fmi_conv_conv, fmi_conv_dwt], dim=1)

        output2 = self.embeding(output1)
        output3 = self.cat_conv(output2)

        output = self.final_layer(torch.cat([output3, image], dim=1))

        return output
