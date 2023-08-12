# How to run

## Train the Autoencoder

```bash
accelerate launch train.py --config_file base/config.yaml
```

## Train the Sensor Data Encoder

Uses the trained decoder from the previous step.

Checkpoints are saved locally.

```bash
accelerate launch train_data_encoder.py --config_file base/config.yaml --unet_checkpoint <path to unet checkpoint>
```