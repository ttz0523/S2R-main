import os
import random

import torch
import torchvision.transforms as transforms
from torch import nn

from network.noise_layers.data.base_dataset import BaseDataset, get_transform
from network.noise_layers.data.image_folder import make_dataset
from PIL import Image


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.phase == "train"
        self.ratio = opt.ratio if self.isTrain else 1.0

        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        if self.isTrain:
            self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(
            make_dataset(self.dir_A, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainA'
        if self.isTrain:
            self.B_paths = sorted(
                make_dataset(self.dir_B, opt.max_dataset_size)
            )  # load images from '/path/to/data/trainB'

        self.A_size = int(len(self.A_paths) * 1)  # get the size of dataset A
        print(self.A_size)
        self.A_paths = self.A_paths[: self.A_size]
        if self.isTrain:
            # self.B_size = int(len(self.B_paths) * (1 - self.ratio))  # get the size of dataset B
            self.B_size = int(len(self.B_paths) * 1)  # get the size of dataset B
            self.B_paths = self.B_paths[-self.B_size:]

        self.BtoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if self.BtoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = (
            self.opt.input_nc if self.BtoA else self.opt.output_nc
        )  # get the number of channels of output image

        if self.BtoA:
            self.transform_A = get_transform(self.opt, self.opt.preprocessB, "B", grayscale=(input_nc == 1))
            if self.isTrain:
                self.transform_B = get_transform(self.opt, self.opt.preprocessA, "A", grayscale=(output_nc == 1))
        else:
            self.transform_A = get_transform(self.opt, self.opt.preprocessA, "A", grayscale=(input_nc == 1))
            if self.isTrain:
                self.transform_B = get_transform(self.opt, self.opt.preprocessB, "B", grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_img = Image.open(A_path).convert("RGB")
        sizeA = A_img.size

        # apply image transformation
        A = self.transform_A(A_img)

        if self.isTrain:
            if self.opt.serial_batches:  # make sure index is within then range
                index_B = index % self.B_size
            else:  # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]

            B_img = Image.open(B_path).convert("RGB")

            # apply image transformation
            B = self.transform_B(B_img)


            return {
                "A": A,
                "B": B,
                "A_paths": A_path,
                "B_paths": B_path,
                "sizeA": sizeA,
            }
        else:
            return {"A": A, "A_paths": A_path, "sizeA": sizeA}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if self.isTrain:
            return max(self.A_size, self.B_size)
        return self.A_size


