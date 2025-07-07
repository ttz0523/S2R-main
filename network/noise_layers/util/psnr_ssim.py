import cv2
import numpy as np
import torch
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from skimage.metrics import structural_similarity


def calculate_psnr(img1, img2, crop_border, input_order="HWC", test_y_channel=False):

    assert img1.shape == img2.shape, f"Image shapes are differnet: {img1.shape}, {img2.shape}."
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are " '"HWC" and "CHW"')
    if type(img1) is torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) is torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    def _psnr(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        max_value = 1.0 if img1.max() <= 1 else 255.0
        return 20.0 * np.log10(max_value / np.sqrt(mse))

    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:, :, :3], img1[:, :, 3:]
        l2, r2 = img2[:, :, :3], img2[:, :, 3:]
        return (_psnr(l1, l2) + _psnr(r1, r2)) / 2
    else:
        return _psnr(img1, img2)


def calculate_psnr_left(img1, img2, crop_border, input_order="HWC", test_y_channel=False):
    assert input_order == "HWC"
    assert crop_border == 0

    img1 = img1[:, 64:, :3]
    img2 = img2[:, 64:, :3]
    return calculate_psnr(img1=img1, img2=img2, crop_border=0, input_order=input_order, test_y_channel=test_y_channel)


def _ssim(img1, img2, max_value):

    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def prepare_for_ssim(img, k):
    import torch

    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k // 2, padding_mode="reflect")
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1.0 / (k * k)

        img = conv(img)

        img = img.squeeze(0).squeeze(0)
        img = img[0::k, 0::k]
    return img.detach().cpu().numpy()


def prepare_for_ssim_rgb(img, k):
    import torch

    with torch.no_grad():
        img = torch.from_numpy(img).float()  # HxWx3

        conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k // 2, padding_mode="reflect")
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1.0 / (k * k)

        new_img = []

        for i in range(3):
            new_img.append(conv(img[:, :, i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)[0::k, 0::k])

    return torch.stack(new_img, dim=2).detach().cpu().numpy()


def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out


def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode="replicate")
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d


def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3

    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()

    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1**2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2**2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1 * img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def _ssim_cly(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    # print(kernel)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window, borderType=bt)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border, input_order="HWC", test_y_channel=False, ssim3d=True):

    assert img1.shape == img2.shape, f"Image shapes are differnet: {img1.shape}, {img2.shape}."
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are " '"HWC" and "CHW"')

    if type(img1) is torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) is torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    def _cal_ssim(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)
            return _ssim_cly(img1[..., 0], img2[..., 0])

        ssims = []

        max_value = 1 if img1.max() <= 1 else 255
        with torch.no_grad():
            final_ssim = _ssim_3d(img1, img2, max_value) if ssim3d else _ssim(img1, img2, max_value)
            ssims.append(final_ssim)


        return np.array(ssims).mean()

    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:, :, :3], img1[:, :, 3:]
        l2, r2 = img2[:, :, :3], img2[:, :, 3:]
        return (_cal_ssim(l1, l2) + _cal_ssim(r1, r2)) / 2
    else:
        return _cal_ssim(img1, img2)


def calculate_ssim_left(img1, img2, crop_border, input_order="HWC", test_y_channel=False, ssim3d=True):
    assert input_order == "HWC"
    assert crop_border == 0

    img1 = img1[:, 64:, :3]
    img2 = img2[:, 64:, :3]
    return calculate_ssim(
        img1=img1, img2=img2, crop_border=0, input_order=input_order, test_y_channel=test_y_channel, ssim3d=ssim3d
    )


def calculate_skimage_ssim(img1, img2):
    return structural_similarity(img1, img2, multichannel=True)


def calculate_skimage_ssim_left(img1, img2):
    img1 = img1[:, 64:, :3]
    img2 = img2[:, 64:, :3]
    return calculate_skimage_ssim(img1=img1, img2=img2)
