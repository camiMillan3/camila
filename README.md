# How to run

## Train the Autoencoder

```bash
accelerate launch train.py --config_file config/base.yml
```

As the latent space is the same for each image input size (1x16x16), the model can be trained on lower 
resolutions and then fine-tuned on higher resolutions.

The pretrained base model is flexible. It can be chosen from https://github.com/qubvel/segmentation_models.pytorch#encoders-
and set in the config file. The depth parameter controls how many layers are taken from the base model.

This link can help choosing a model: https://paperswithcode.com/lib/timm

## Train the Sensor Data Encoder

Uses the trained decoder from the previous step.

Checkpoints are saved locally.

```bash
accelerate launch train_data_encoder.py --config_file config/base.yml --unet_checkpoint <path to unet checkpoint>
```

