from torch.utils.data import DataLoader

from utils import hid_utils
from utils import *
from network.Network import *

from utils.load_train_setting import *

'''
train
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Network(H, W, message_length, noise_layers, device, batch_size, lr, is_differentiable, epoch_number)

train_dataset = TDataset(os.path.join(dataset_path, "train"), H, W)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = TDataset(os.path.join(dataset_path, "validation"), H, W)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

if train_continue:
	EC_path = "results/" + train_continue_path + "/models/EC_" + str(train_continue_epoch) + ".pth"
	D_path = "results/" + train_continue_path + "/models/D_" + str(train_continue_epoch) + ".pth"
	network.load_model(EC_path, D_path)

print("\nStart training : \n\n")
start = 1
for epoch in range(1, epoch_number+1):

	epoch += train_continue_epoch if train_continue else 0

	running_result = {
		"error_rate": 0.0,
		"psnr": 0.0,
		"ssim": 0.0,
		"g_loss": 0.0,
		"g_loss_on_discriminator": 0.0,
		"g_loss_on_encoder": 0.0,
		"g_loss_on_decoder": 0.0,
		"d_cover_loss": 0.0,
		"d_encoded_loss": 0.0,
		"g_loss_on_perceptual": 0.0,
		"scheduler_encoder_decoder_lr": 0.0,
		"scheduler_discriminator_lr": 0.0,
	}

	start_time = time.time()

	'''
	train
	'''
	num = 0
	for i, images, in enumerate(train_dataloader):
		image = images.to(device)
		message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

		result = network.train(image, message)

		print('----------------------------------------------------')
		print('Epoch: {}/{} Step: {}'.format(epoch, epoch_number, i))

		for key in result:
			running_result[key] += float(result[key])
			print(key+'='+str(float(result[key])))
		num += 1
	# update lr
	network.scheduler_encoder_decoder.step()
	network.scheduler_discriminator.step()
	'''
	train results
	'''
	content = "Epoch " + str(epoch) + " time: " + str(int(time.time() - start_time)) + "\n"


	for key in running_result:
		content += key + "=" + str(running_result[key] / num) + ","
	content += "\n"

	print('-------------------------------------------------------')
	print("train")
	print(content)

	with open(result_folder + "/train_log.txt", "a") as file:
		file.write(content)
	print(content)
	train_duration = time.time() - start_time
	hid_utils.write_losses(os.path.join(result_folder, 'train.csv'), running_result, epoch, train_duration, num, start)
	'''
	validation
	'''

	val_result = {
		"error_rate": 0.0,
		"psnr": 0.0,
		"ssim": 0.0,
		"g_loss": 0.0,
		"g_loss_on_discriminator": 0.0,
		"g_loss_on_encoder": 0.0,
		"g_loss_on_decoder": 0.0,
		"d_cover_loss": 0.0,
		"d_encoded_loss": 0.0,
		"g_loss_on_perceptual": 0.0,
	}

	start_time = time.time()

	saved_iterations = np.random.choice(np.arange(len(val_dataset)), size=save_images_number, replace=False)
	saved_all = None

	num = 0
	for i, images in enumerate(val_dataloader):
		image = images.to(device)
		message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

		result, (images, encoded_images, noised_images, messages, decoded_messages) = network.validation(image, message)

		print('----------------------------------------------------')
		print('Epoch: {}/{} Step: {}'.format(epoch, epoch_number, i))

		for key in result:
			val_result[key] += float(result[key])
			print(key+'='+str(float(result[key])))
		num += 1

		if i in range(4):
			if saved_all is None:
				saved_all = get_random_images(image, encoded_images, noised_images)
			else:
				saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)
	if saved_all is not None:
		save_images(saved_all, epoch, result_folder + "images/", resize_to=(W, H))
	else:
		print('there is no images in saved_all!')

	'''
	validation results
	'''
	content = "Epoch " + str(epoch) + " time: " + str(int(time.time() - start_time)) + "\n"
	for key in val_result:
		content += key + "=" + str(val_result[key] / num) + ","
	content += "\n"

	print('-------------------------------------------------------')
	print("validation")
	print(content)

	with open(result_folder + "/val_log.txt", "a") as file:
		file.write(content)
	print(content)
	train_duration = time.time() - start_time
	hid_utils.write_losses(os.path.join(result_folder, 'val.csv'), val_result, epoch, train_duration, num, start)
	start = 0
	'''
	save model
	'''
	path_model = result_folder + "models/"
	path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
	path_discriminator = path_model + "D_" + str(epoch) + ".pth"
	network.save_model(path_encoder_decoder, path_discriminator)
