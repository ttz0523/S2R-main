from .settings import *
import os
import time

'''
params setting
'''
filename = "en_de_test_settings.json"
settings = JsonConfig()
settings.load_json_file(filename)

is_differentiable = settings.is_differentiable
is_encode = settings.is_encode
with_diffusion = settings.with_diffusion

Ico_dataset_path = settings.Ico_dataset_path
Ien_path = settings.Ien_path
M_txt_path = settings.M_txt_path

batch_size = 1
model_epoch = settings.model_epoch
strength_factor = settings.strength_factor
save_images_number = settings.save_images_number
lr = 1e-3
H, W, message_length = settings.H, settings.W, settings.message_length
noise_layers = settings.noise_layers

result_folder = "results/" + settings.result_folder
test_base = "/test_"
for layer in settings.noise_layers:
	test_base += layer + "_"
test_param = result_folder + test_base + "s{}_params.json".format(strength_factor)
test_en_log = result_folder + test_base + "s{}_en_log.txt".format(strength_factor)
test_de_log = result_folder + test_base + "s{}_de_log.txt".format(strength_factor)

if is_encode:
	with open(test_en_log, "w") as file:
		print("")
if not is_encode:
	with open(test_de_log, "w") as file:
		print("")

with open(test_param, "w") as file:
	content = ""
	for item in settings.get_items():
		content += item[0] + " = " + str(item[1]) + "\n"
	print(content)

	with open(filename, "r") as setting_file:
		content = setting_file.read()
		file.write(content)

#  生成txt标签文件
if is_encode:
	txt_log = result_folder + "test_de.txt"
	with open(txt_log, "w") as file:
		print("")

