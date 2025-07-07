import torch

from .gan_loss import GANLoss
from .lpips_loss import LPIPS
from .vgg_loss import VGGLoss

__all__ = ['GANLoss', 'LPIPS', 'VGGLoss']
# pylint: disable=too-many-arguments
# pylint: disable=redefined-builtin


def cal_gradient_penalty(netD, real_data, fake_data, device, lambda_gp=0.1, type="mixed", constant=1.0):
    """Calculate the gradient penalty loss, used in WGAN-GP

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- torch device
        type (str)                  -- if we mix real and fake data or not
            Choices: [real | fake | mixed].
        constant (float)            -- the constant used in formula:
            (||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp == 0.0:
        return 0.0, None

    if type == "real":
        interpolatesv = real_data
    elif type == "fake":
        interpolatesv = fake_data
    elif type == "mixed":
        alpha = torch.rand(real_data.shape[0], 1, device=device)
        alpha = (
            alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0])
            .contiguous()
            .view(*real_data.shape)
        )

        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError("{} not implemented".format(type))

    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolatesv,
        grad_outputs=[torch.ones(disc_interpolates[idx].size()).to(device) for idx in range(1)],
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )

    gradients = gradients[0].view(real_data.size(0), -1)

    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp

    return gradient_penalty, gradients
