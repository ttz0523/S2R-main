import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def weights_init(model):
    """init from article"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 0.1)
            nn.init.constant_(m.bias, 0)


def calc_gram(x):
    (n, c, h, w) = x.size()
    f = x.view(n, c, w * h)
    f_trans = f.transpose(1, 2)
    gram = f.bmm(f_trans) / (c * h * w)
    return gram


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calc_ssim(img1, img2, window_size=11):
    """calculate SSIM"""
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average=True)


def calc_psnr(img1, img2):
    """calculate PNSR on cuda and cpu: img1 and img2 have range [0, 255]"""
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def rgb2lum(arr):
    small = np.where(arr <= 0.04045)
    big = np.where(arr > 0.04045)
    arr[small] /= 12.92
    arr[big] = ((arr[big] + 0.055) / 1.055) ** 2.4
    return arr


def lum(image):
    """
    turn BGR to Lum
    :param image: image in sRGB area, range 255
    :return: image in Lum
    """
    assert image.shape[0] == 3, "make sure the layout is (c, h, w), BGR"
    _, h, w = image.shape
    image = image.astype(np.float)
    v_b = image[0, ...] / 255
    v_g = image[1, ...] / 255
    v_r = image[2, ...] / 255
    print(rgb2lum(v_r))
    l_image = 0.2126 * rgb2lum(v_r) + 0.7152 * rgb2lum(v_g) + 0.0722 * rgb2lum(v_b)
    return l_image


def upsampling(img, x, y):
    func = nn.Upsample(size=[x, y], mode='bilinear', align_corners=True)
    return func(img)


def generate_noise(size, channels=1, type='gaussian', scale=2, noise=None):
    if type == 'gaussian':
        noise = torch.randn(channels, size[0], round(size[1]/scale), round(size[2]/scale))
        noise = upsampling(noise, size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(channels, size[0], size[1], size[2]) + 5
        noise2 = torch.randn(channels, size[0], size[1], size[2])
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(channels, size[0], size[1], size[2])
    return noise * 10.


def concat_noise(img, *args):
    noise = generate_noise(*args)
    if isinstance(img, torch.Tensor):
        noise = noise.to(img.device)
    else:
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    mixed_img = torch.cat((img, noise), 1)
    return mixed_img


class ImageEvaluation(object):
    def __init__(self, img, mode):
        super(ImageEvaluation, self).__init__()
        self.img = img
