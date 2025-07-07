import kornia.losses
from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio

from .loss.perceptual import ResNetPL
from .warmup_scheduler import GradualWarmupScheduler


class Network:

    def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, is_differentiable=False, epoch_number=150):
        # device
        self.device = device

        # network
        self.encoder_decoder = EncoderDecoder(H, W, message_length, noise_layers, is_differentiable).to(device)

        self.discriminator = Discriminator().to(device)

        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator = torch.nn.DataParallel(self.discriminator)

        # mark "cover" as 1, "encoded" as 0
        self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
        self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

        # Optimizer
        print(lr)
        self.opt_encoder_decoder = optim.AdamW(self.encoder_decoder.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                               weight_decay=0.02)
        self.opt_discriminator = optim.AdamW(self.discriminator.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                             weight_decay=0.02)

        # Scheduler
        print("Using warmup and cosine strategy!")
        nepoch = epoch_number
        warmup_epochs = 3
        self.scheduler_cosine_encoder_decoder = optim.lr_scheduler.CosineAnnealingLR(self.opt_encoder_decoder,
                                                                                     nepoch - warmup_epochs,
                                                                                     eta_min=1e-6)
        self.scheduler_encoder_decoder = GradualWarmupScheduler(self.opt_encoder_decoder, multiplier=1,
                                                                total_epoch=warmup_epochs,
                                                                after_scheduler=self.scheduler_cosine_encoder_decoder)
        self.scheduler_cosine_discriminator = optim.lr_scheduler.CosineAnnealingLR(self.opt_discriminator,
                                                                                   nepoch - warmup_epochs, eta_min=1e-6)
        self.scheduler_discriminator = GradualWarmupScheduler(self.opt_discriminator, multiplier=1,
                                                              total_epoch=warmup_epochs,
                                                              after_scheduler=self.scheduler_cosine_discriminator)

        self.scheduler_encoder_decoder.step()
        self.scheduler_discriminator.step()

        # loss function
        self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
        self.criterion_MSE = nn.MSELoss().to(device)
        self.perceptual_loss = self.loss_resnet_pl = ResNetPL(weight=30,
                                                              weights_path='network/loss').to(device)

        # weight of encoder-decoder loss
        self.discriminator_weight = 0.0001
        self.encoder_weight = 1
        self.decoder_weight = 10
        self.perceptual_weight = 0.01

    def train(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.train()
        self.discriminator.train()

        with torch.enable_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(
                images, messages)

            '''
			train discriminator
			'''
            self.opt_discriminator.zero_grad()

            # RAW : target label for image should be "cover"(1)
            d_label_cover = self.discriminator(images)
            d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
            d_cover_loss.backward()

            # GAN : target label for encoded image should be "encoded"(0)
            d_label_encoded = self.discriminator(encoded_images.detach())
            d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
            d_encoded_loss.backward()

            self.opt_discriminator.step()

            '''
			train encoder and decoder
			'''
            self.opt_encoder_decoder.zero_grad()
            # Perceptual
            g_loss_on_perceptual = self.perceptual_loss(encoded_images, images)
            # GAN : target label for encoded image should be "cover"(0)
            g_label_decoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

            # RESULT : the decoded message should be similar to the raw message
            g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

            # full loss
            g_loss = g_loss_on_discriminator * self.discriminator_weight + self.encoder_weight * g_loss_on_encoder + \
                     self.decoder_weight * g_loss_on_decoder + self.perceptual_weight * g_loss_on_perceptual

            g_loss.backward()
            self.opt_encoder_decoder.step()

            # psnr
            # psnr = -kornia.losses.psnr_loss(encoded_images.detach(), images, 255)
            encoded_images_np = encoded_images.detach().cpu().numpy()
            images_np = images.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(encoded_images_np, images_np, data_range=2)
            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")

            # print lr
            scheduler_encoder_decoder_lr = self.scheduler_encoder_decoder.get_lr()[0]
            scheduler_discriminator_lr = self.scheduler_discriminator.get_lr()[0]
        '''
		decoded message error rate
		'''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder": g_loss_on_encoder,
            "g_loss_on_decoder": g_loss_on_decoder,
            "d_cover_loss": d_cover_loss,
            "d_encoded_loss": d_encoded_loss,
            "g_loss_on_perceptual": g_loss_on_perceptual,
            "scheduler_encoder_decoder_lr": scheduler_encoder_decoder_lr,
            "scheduler_discriminator_lr": scheduler_discriminator_lr,
        }
        return result


    def validation(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.eval()
        self.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(
                images, messages)

            '''
			validate discriminator
			'''
            # Perceptual
            g_loss_on_perceptual = self.perceptual_loss(encoded_images, images)
            # RAW : target label for image should be "cover"(1)
            d_label_cover = self.discriminator(images)
            d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

            # GAN : target label for encoded image should be "encoded"(0)
            d_label_encoded = self.discriminator(encoded_images.detach())
            d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])

            '''
			validate encoder and decoder
			'''

            # GAN : target label for encoded image should be "cover"(0)
            g_label_decoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

            # RESULT : the decoded message should be similar to the raw message
            g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

            # full loss
            g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
                     self.decoder_weight * g_loss_on_decoder + self.perceptual_weight * g_loss_on_perceptual

            # psnr
            psnr = -kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")

        '''
		decoded message error rate
		'''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder": g_loss_on_encoder,
            "g_loss_on_decoder": g_loss_on_decoder,
            "d_cover_loss": d_cover_loss,
            "d_encoded_loss": d_encoded_loss,
            "g_loss_on_perceptual": g_loss_on_perceptual,
        }

        return result, (images, encoded_images, noised_images, messages, decoded_messages)

    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[0]

        message = message.gt(0.5)
        decoded_message = decoded_message.gt(0.5)
        error_rate = float(sum(message != decoded_message)) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages):
        error_rate = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
        error_rate /= batch_size
        return error_rate

    def save_model(self, path_encoder_decoder: str, path_discriminator: str):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator.module.state_dict(), path_discriminator)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder), strict=False)

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator), strict=False)
