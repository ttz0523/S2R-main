# S2R

Official PyTorch implementation of the AAAI 2026 paper "Sim-to-Real: An Unsupervised Noise Layer for Screen-Camera Watermarking Robustness."

This repository implements a S2R image watermarking framework. It supports model training, watermark encoding, physical shooting simulation, geometric correction, and message decoding.

## Training

Use `train.py` to train the model. Configuration is defined in `train_setting.json`:

```json
{
  "project_name": "Base-main",
  "is_differentiable": true,
  "epoch_number": 150,
  "batch_size": 16,
  "train_continue": false,
  "train_continue_path": "/",
  "train_continue_epoch": 44,
  "dataset_path": "datasets_mini/",
  "save_images_number": 2,
  "H": 128,
  "W": 128,
  "message_length": 64,
  "lr": 1e-3,
  "noise_layers": [
    "SCNG()"
  ]
}
```

To start training:
```bash
python train.py
```

## 2. Encoding & Decoding
Script: `en_de_test.py`

Configuration: `en_de_test_setting.json`

### Encoding
Use the following configuration to embed messages into images (default is noise-free; please implement custom noise injection if needed):
```json
{
  "result_folder": "SCNG_test/",
  "model_epoch": 150,
  "is_differentiable": true,
  "is_encode": true,
  "with_diffusion": false,
  "strength_factor": 1.0,
  "Ico_dataset_path": "testsets_100/",
  "Ien_path": "results/SCNG_test/test_pes",
  "M_txt_path": "results/SCNG_test/test_de.txt",
  "H": 128,
  "W": 128,
  "message_length": 64,
  "save_images_number": 1,
  "noise_layers": [
    "Identity()"
  ]
}
```
Run:

```bash
python en_de_test.py
```

## 3. Capture & Cropping
After encoding, display or print the watermarked images from `test_en/` and capture them with a camera or mobile device.
Save the captured photos to the `test_sc/` directory.

Then use:

```bash
python PespectiveTransformation.py
```
The perspective-corrected images will be saved to `test_pes/`.

## 4. Decoding
Use the following configuration to decode from the captured and corrected images:

```json
{
  "result_folder": "SCNG_test/",
  "model_epoch": 150,
  "is_differentiable": true,
  "is_encode": false,
  "with_diffusion": false,
  "strength_factor": 1.0,
  "Ico_dataset_path": "testsets_100/",
  "Ien_path": "results/SCNG_test/test_pes",
  "M_txt_path": "results/SCNG_test/test_de.txt",
  "H": 128,
  "W": 128,
  "message_length": 64,
  "save_images_number": 1,
  "noise_layers": [
    "Identity()"
  ]
}
```

Then run:

```bash
python en_de_test.py
```
