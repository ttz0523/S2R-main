import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.backbone.arch_util import LayerNorm2d
from models.backbone.local_arch import Local_Base



class G_Unet_add_input(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        nz,
        num_downs,
        ngf=64,
        norm_layer=None,
        nl_layer=None,
        use_dropout=False,
        upsample="basic",
    ):
        super(G_Unet_add_input, self).__init__()
        self.nz = nz
        max_nchn = 8
        # construct unet structure
        unet_block = UnetBlock(
            ngf * max_nchn,
            ngf * max_nchn,
            ngf * max_nchn,
            innermost=True,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            upsample=upsample,
        )
        for i in range(num_downs - 5):
            unet_block = UnetBlock(
                ngf * max_nchn,
                ngf * max_nchn,
                ngf * max_nchn,
                unet_block,
                norm_layer=norm_layer,
                nl_layer=nl_layer,
                use_dropout=use_dropout,
                upsample=upsample,
            )
        unet_block = UnetBlock(
            ngf * 4, ngf * 4, ngf * max_nchn, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample
        )
        unet_block = UnetBlock(
            ngf * 2, ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample
        )
        unet_block = UnetBlock(
            ngf, ngf, ngf * 2, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample
        )
        unet_block = UnetBlock(
            input_nc + nz,
            output_nc,
            ngf,
            unet_block,
            outermost=True,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            upsample=upsample,
        )

        self.model = unet_block

    def forward(self, x, z=None):
        if self.nz > 0:
            # z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
            #     z.size(0), z.size(1), x.size(2), x.size(3))
            z_img = F.interpolate(z, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z

        return self.model(x_with_z)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(
        self,
        input_nc,
        outer_nc,
        inner_nc,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=None,
        nl_layer=None,
        use_dropout=False,
        upsample="basic",
        padding_type="zero",
    ):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == "reflect":
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


def upsampleLayer(inplanes, outplanes, upsample="basic", padding_type="zero"):
    # padding_type = 'zero'
    if upsample == "basic":
        upconv = [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == "bilinear":
        upconv = [
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0),
        ]
    else:
        raise NotImplementedError("upsample layer [%s] not implemented" % upsample)
    return upconv


###################################################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        nz,
        num_downs,
        ngf=64,
        norm_layer=None,
        nl_layer=None,
        use_dropout=False,
        upsample="basic",
    ):
        super(G_Unet_add_all, self).__init__()
        self.nz = nz
        # construct unet structure
        unet_block = UnetBlock_with_z(
            ngf * 8,
            ngf * 8,
            ngf * 8,
            nz,
            None,
            innermost=True,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            upsample=upsample,
        )
        unet_block = UnetBlock_with_z(
            ngf * 8,
            ngf * 8,
            ngf * 8,
            nz,
            unet_block,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            use_dropout=use_dropout,
            upsample=upsample,
        )
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(
                ngf * 8,
                ngf * 8,
                ngf * 8,
                nz,
                unet_block,
                norm_layer=norm_layer,
                nl_layer=nl_layer,
                use_dropout=use_dropout,
                upsample=upsample,
            )
        unet_block = UnetBlock_with_z(
            ngf * 4, ngf * 4, ngf * 8, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample
        )
        unet_block = UnetBlock_with_z(
            ngf * 2, ngf * 2, ngf * 4, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample
        )
        unet_block = UnetBlock_with_z(
            ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample
        )
        unet_block = UnetBlock_with_z(
            input_nc,
            output_nc,
            ngf,
            nz,
            unet_block,
            outermost=True,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            upsample=upsample,
        )
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z(nn.Module):
    def __init__(
        self,
        input_nc,
        outer_nc,
        inner_nc,
        nz=0,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=None,
        nl_layer=None,
        use_dropout=False,
        upsample="basic",
        padding_type="zero",
    ):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == "reflect":
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        # print(x.size())
        if self.nz > 0:
            # z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            # breakpoint()
            z_img = F.interpolate(z, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


###################################################################


class G_Unet_add_middle(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        nz,
        num_downs,
        ngf=64,
        norm_layer=None,
        nl_layer=None,
        use_dropout=False,
        upsample="basic",
    ):
        super(G_Unet_add_middle, self).__init__()
        lrelu = nn.LeakyReLU(negative_slope=0.1)
        front_RBs = 10
        back_RBs = 20
        num_image_channels = input_nc
        nf = ngf

        # Features extraction
        resBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        feature_extractor = []

        feature_extractor.append(nn.Conv2d(num_image_channels, nf, 3, 1, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)

        for i in range(front_RBs):
            feature_extractor.append(resBlock_noBN_f())

        self.feature_extractor = nn.Sequential(*feature_extractor)

        # Kernel extractor
        # self.kernel_extractor = KernelExtractor(opt)

        # kernel adapter
        self.adapter = KernelAdapter(
            ngf,
            ngf,
            nz,
            num_downs,
            ngf,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            use_dropout=use_dropout,
            upsample=upsample,
        )

        # Reconstruction
        recon_trunk = []
        for i in range(back_RBs):
            recon_trunk.append(resBlock_noBN_f())

        # upsampling
        recon_trunk.append(nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True))
        recon_trunk.append(nn.PixelShuffle(2))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True))
        recon_trunk.append(nn.PixelShuffle(2))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(64, num_image_channels, 3, 1, 1, bias=True))

        self.recon_trunk = nn.Sequential(*recon_trunk)

    def forward(self, x, kernel):
        # breakpoint()
        B, C, H, W = x.shape
        base = x

        x = self.feature_extractor(x)

        out = self.adapter(x, kernel)
        out = self.recon_trunk(out)
        out += base

        return out


# The function G in the paper
class KernelAdapter(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        nz,
        num_downs,
        ngf=64,
        norm_layer=None,
        nl_layer=None,
        use_dropout=False,
        upsample="basic",
    ):
        super(KernelAdapter, self).__init__()
        # input_nc = opt["nf"]
        # output_nc = opt["nf"]
        # ngf = opt["nf"]
        # norm_layer = arch_util.get_norm_layer(opt["Adapter"]["norm"])

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, x, k):
        """Standard forward"""
        # breakpoint()
        return self.model(x, k)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) --previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) is functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            up = [uprelu, upconv, nn.Tanh()]
            down = [downconv]
            self.down = nn.Sequential(*down)
            self.submodule = submodule
            self.up = nn.Sequential(*up)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            self.down = nn.Sequential(*down)
            self.up = nn.Sequential(*up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            # upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # upconv = DoubleConv(inner_nc * 2, outer_nc)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up += [nn.Dropout(0.5)]

            self.down = nn.Sequential(*down)
            self.submodule = submodule
            self.up = nn.Sequential(*up)

    def forward(self, x, noise):

        if self.outermost:
            return self.up(self.submodule(self.down(x), noise))
        elif self.innermost:  # add skip connections
            if noise is None:
                noise = torch.randn((1, 512, 8, 8)).cuda() * 0.0007
            # print("Noise = ",noise.shape)
            # print("X = ",x.shape)
            return torch.cat((self.up(torch.cat((self.down(x), noise), dim=1)), x), dim=1)
        else:
            return torch.cat((self.up(self.submodule(self.down(x), noise)), x), dim=1)


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=False)
        out = self.conv2(out)
        return identity + out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


######################################################################


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.ending = nn.Conv2d(
            in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
