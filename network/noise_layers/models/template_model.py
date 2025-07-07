import torch

from . import networks
from .base_model import BaseModel


class TemplateModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(
            dataset_mode="aligned"
        )
        if is_train:
            parser.add_argument(
                "--lambda_regression", type=float, default=1.0, help="weight for the regression loss"
            )

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        self.loss_names = ["loss_G"]

        self.visual_names = ["data_A", "data_B", "output"]

        self.model_names = ["G"]

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        if self.isTrain:

            self.criterionLoss = torch.nn.L1Loss()

            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

    def set_input(self, input):

        AtoB = self.opt.direction == "AtoB"  # use <direction> to swap data_A and data_B
        self.data_A = input["A" if AtoB else "B"].to(self.device)  # get image data A
        self.data_B = input["B" if AtoB else "A"].to(self.device)  # get image data B
        self.image_paths = input["A_paths" if AtoB else "B_paths"]  # get image paths

    def forward(self):

        self.output = self.netG(self.data_A)  # generate output image given the input data_A

    def backward(self):

        self.loss_G = self.criterionLoss(self.output, self.data_B) * self.opt.lambda_regression
        self.loss_G.backward()  # calculate gradients of network G w.r.t. loss_G

    def optimize_parameters(self):

        self.forward()  # first call forward to calculate intermediate results
        self.optimizer.zero_grad()  # clear network G's existing gradients
        self.backward()  # calculate gradients for network G
        self.optimizer.step()  # update gradients for network G
