import torch
import torch.nn as nn
from .dwt2d import DWTForward, DWTInverse
from .. import ConvBNRelu


class DWTBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWTBlock, self).__init__()
        self.first_conv = ConvBNRelu(in_channels, in_channels)
        self.dwt_forward = DWTForward(J=1, wave='db1', mode='zero')
        self.ll_conv = ConvBNRelu(in_channels, in_channels)
        self.lh_conv = ConvBNRelu(in_channels, in_channels)
        self.hl_conv = ConvBNRelu(in_channels, in_channels)
        self.hh_conv = ConvBNRelu(in_channels, in_channels)
        self.dwt_inverse = DWTInverse(wave='db1', mode='zero')
        self.last_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, image):
        image_before_dwt = self.first_conv(image)
        x = image_before_dwt
        Yl, Yh = self.dwt_forward(image_before_dwt)  # Yl为低频子带, Yh为三个高频子带
        Yl = self.ll_conv(Yl)
        LH = Yh[0][:, :, 0]
        HL = Yh[0][:, :, 1]
        HH = Yh[0][:, :, 2]
        LH = self.lh_conv(LH)
        HL = self.hl_conv(HL)
        HH = self.hh_conv(HH)
        with torch.no_grad():
            Yh[0][:, :, 0] = LH
            Yh[0][:, :, 1] = HL
            Yh[0][:, :, 2] = HH
        image_after_dwt = self.dwt_inverse((Yl, Yh))  # 逆DWT
        image_after_dwt = image_after_dwt + x
        output = self.last_conv(image_after_dwt)
        return output, Yl, LH, HL, HH
