import random
from . import *
from .Encoder_MP import Encoder_MP
from .Decoder import Decoder
from .Noise import Noise


class EncoderDecoder(nn.Module):
    '''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

    def __init__(self, H, W, message_length, noise_layers, is_differentiable):
        super(EncoderDecoder, self).__init__()
        self.is_differentiable = is_differentiable
        self.encoder = Encoder_MP(H, W, message_length)
        self.noise = Noise(noise_layers)
        self.decoder = Decoder(H, W, message_length)

    def forward(self, image, message):
        if self.is_differentiable is True:
            encoded_image = self.encoder(image, message)
            noised_image = self.noise([encoded_image, image])
            decoded_message = self.decoder(noised_image)
            return encoded_image, noised_image, decoded_message

        else:
            b = random.getrandbits(1), bool(random.getrandbits(1))
            if b:
                encoded_image = self.encoder(image, message)
                with torch.no_grad():
                    noised_image = self.noise([encoded_image, image])
                    images_gap = noised_image - encoded_image
                    images_gap = images_gap.detach()
                noised_image = encoded_image + images_gap
                decoded_message = self.decoder(noised_image)
                return encoded_image, noised_image, decoded_message
            else:
                encoded_image = self.encoder(image, message)
                noised_image = self.noise([encoded_image, image])
                with torch.no_grad():
                    images_gap = noised_image - encoded_image
                    images_gap = images_gap.detach()
                noised_image = noised_image - images_gap
                decoded_message = self.decoder(noised_image)
                return encoded_image, noised_image, decoded_message
