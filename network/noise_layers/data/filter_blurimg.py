import os
import pickle
from os.path import join as osp

import cv2
import numpy as np
from tqdm import tqdm
from util.psnr_ssim import to_y_channel


def reorder_image(img, input_order="HWC"):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ["HWC", "CHW"]:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are " "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    return img


def calculate_psnr(img1, img2, crop_border, input_order="HWC", test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).



    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, f"Image shapes are differnet: {img1.shape}, {img2.shape}."
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are " '"HWC" and "CHW"')
    # if type(img1) == torch.Tensor:
    #     if len(img1.shape) == 4:
    #         img1 = img1.squeeze(0)
    #     img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    # if type(img2) == torch.Tensor:
    #     if len(img2.shape) == 4:
    #         img2 = img2.squeeze(0)
    #     img2 = img2.detach().cpu().numpy().transpose(1,2,0)

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


def psnr_pair(s1, s2):
    img1 = cv2.imread(s1)
    img2 = cv2.imread(s2)
    return calculate_psnr(img1, img2, 0)


sharp_path = "/home/dangpb1/Research/datasets/RSBlur/RSBlur-b2b/train_sharp"
blur_path = "/home/dangpb1/Research/datasets/RSBlur/RSBlur-b2b/trainA"
files = sorted(os.listdir(sharp_path))

psnr_score = {}
for each in tqdm(files):
    psnr_score[each] = psnr_pair(osp(sharp_path, each), osp(blur_path, each))
sorted_psnr = sorted(psnr_score.items(), key=lambda x: x[1], reverse=True)
handle = open("rsblur.pkl", "wb")
pickle.dump(sorted_psnr, handle, protocol=pickle.HIGHEST_PROTOCOL)
# breakpoint()
