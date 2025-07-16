from PIL import Image
from torch.utils.data import DataLoader

from network.Network import *
from utils import *
from utils.load_en_de_test_setting import *

'''
test
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion, is_differentiable)
EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
print(EC_path)
network.load_model_ed(EC_path)

if is_encode:
    test_dataset = Test_Dataset(Ico_dataset_path, H, W)

else:
    test_dataset = Img_Mes_Dataset(Ien_path, M_txt_path, H, W)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
print("\nStart Testing : \n\n")

start_time = time.time()

saved_all = None

if is_encode:
    test_result = {
        "error_rate": 0.0,
        "psnr": 0.0,
        "ssim": 0.0
    }
    num = 0
    for i, images in enumerate(test_dataloader):
        image = images.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
        '''
        test
        '''
        network.encoder_decoder.eval()
        network.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(network.device), message.to(network.device)

            encoded_images = network.encoder_decoder.module.encoder(images,
                                                                                                                messages)
            encoded_images = images + (encoded_images - image) * strength_factor
            noised_images = network.encoder_decoder.module.noise([encoded_images, images])

            decoded_messages = network.encoder_decoder.module.decoder(noised_images)

            # psnr
            psnr = -kornia.losses.psnr_loss(encoded_images.detach(), images, 2).item()

            # ssim
            ssim = (1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5,
                                                    reduction="mean")).item()

        '''
        decoded message error rate
        '''
        error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
        }

        for key in result:
            test_result[key] += float(result[key])

        '''
        test results
        '''
        content = "Image " + str(i + 1) + ".\nMessage: " + np.array_str(message.cpu().numpy()) + " : \n"
        for key in test_result:
            content += key + "=" + str(result[key]) + ","
        content += "\n"

        with open(test_en_log, "a") as file:
            file.write(content)

        if not os.path.exists(result_folder + "test_en/"):
            os.mkdir(result_folder + "test_en/")

        filename = os.path.join(result_folder + "test_en", 'encoded_{}.png'.format(num))
        file_position = os.path.join(result_folder + "test_en", 'encoded_{}.png'.format(num))

        encoded_images = encoded_images[:encoded_images.shape[0], :, :, :].cpu()
        encoded_images = (encoded_images + 1) / 2
        encoded_images = F.interpolate(encoded_images, size=(W, H))
        encoded_images = encoded_images.unsqueeze(0)
        shape = encoded_images.shape
        encoded_images = encoded_images.permute(0, 3, 1, 4, 2).reshape(shape[3] * shape[0], shape[4] * shape[1],
                                                                       shape[2])
        encoded_images = encoded_images.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()

        saved_image = Image.fromarray(np.array(encoded_images, dtype=np.uint8)).convert("RGB")
        saved_image.save(file_position)

        print("image({}) has been saved in ".format(num), file_position)

        with open(txt_log, "a") as file:
            txt_content = os.path.basename(filename) + '|' + np.array_str(message[0].cpu().numpy())[1:-2].replace("\n",
                                                                                                                  "") + '\n'
            file.write(txt_content)

        print(content)
        print("------------------------------------------------------------------------------------------------")
        num += 1

    '''
    test results
    '''
    content = "Average : \n"
    for key in test_result:
        content += key + "=" + str(test_result[key] / num) + ","
    content += "\n"

    with open(test_en_log, "a") as file:
        file.write(content)

    print(content)


elif not is_encode:
    num = 0
    test_result = {
        "error_rate": 0.0,
    }
    for i, (images, messages) in enumerate(test_dataloader):

        image = images.to(device)

        messages = messages[0]
        messages = np.array(messages.split(". "), dtype=float)
        message = torch.Tensor(messages).unsqueeze(0).repeat(image.shape[0], 1).to(device)
        '''
        test
        '''
        network.encoder_decoder.eval()
        network.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(network.device), message.to(network.device)

            noised_images = images

            decoded_messages = network.encoder_decoder.module.decoder(noised_images)

        '''
        decoded message error rate
        '''
        error_rate = network.decoded_message_error_rate_batch(message, decoded_messages)

        result = {
            "error_rate": error_rate,
        }

        for key in result:
            test_result[key] += float(result[key])

        '''
        test results
        '''
        content = "Image " + str(i + 1) + " .\nMessage: " + np.array_str(
            message.cpu().numpy()) + " .\nDecoded_message: " + np.array_str(decoded_messages.cpu().numpy()) + " . \n"
        for key in test_result:
            content += key + "=" + str(result[key]) + ","
        content += "\n"

        with open(test_de_log, "a") as file:
            file.write(content)

        print(content)
        print("------------------------------------------------------------------------------------------------")
        num += 1

    '''
    test results
    '''
    content = "Average : \n"
    for key in test_result:
        content += key + "=" + str(test_result[key] / num) + ","
    content += "\n"
    with open(test_de_log, "a") as file:
        file.write(content)

    print(content)
