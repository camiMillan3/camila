# How to run

## Train the Autoencoder

```bash
accelerate launch train.py --config_file base/config.yaml
```

As the latent space is the same for each image input size (1x16x16), the model can be trained on lower 
resolutions and then fine-tuned on higher resolutions.

## Train the Sensor Data Encoder

Uses the trained decoder from the previous step.

Checkpoints are saved locally.

```bash
accelerate launch train_data_encoder.py --config_file base/config.yaml --unet_checkpoint <path to unet checkpoint>
```

