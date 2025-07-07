import os

import torch
from math import log10
from torchvision.transforms import Resize
import torch.nn.functional as F
from torch import nn

from network.noise_layers.models import create_model
from skimage.metrics import structural_similarity as ssim

from network.noise_layers.options.test_options import TestOptions

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def metrics(img1, img2):
    img1_np = img1.cpu().detach().numpy()
    img2_np = img2.cpu().detach().numpy()

    # Calculate PSNR using PyTorch
    mse = F.mse_loss(img1, img2).item()
    psnr = 10 * log10(1 / mse)

    # Calculate SSIM using scikit-image
    ssim_val = ssim(img1_np, img2_np, multichannel=True)
    return psnr, ssim_val

class SCNG(nn.Module):

    def __init__(self):
        super(SCNG, self).__init__()
        opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        opt.num_threads = 0  # test code only supports num_threads = 0
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = (
            True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        )
        opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        if opt.max_dataset_size == -1:
            opt.max_dataset_size = float("inf")

        self.model = create_model(opt)  # create a model given opt.model and other options
        self.model.setup(opt)  # regular setup: load and print networks; create schedulers

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        self.model.eval()

        image = (image + 1) / 2

        resize_transform_to256 = Resize(size=256)
        resize_transform_back = Resize(size=image.shape[2])

        image = resize_transform_to256(image)

        embed_image = {"A": image}
        self.model.set_input(embed_image)  # unpack data from data loader

        fake_B_ = self.model.forward()  # run inference
        fake_B_ = fake_B_ * 2 - 1

        fake_B_ = resize_transform_back(fake_B_)

        return fake_B_
