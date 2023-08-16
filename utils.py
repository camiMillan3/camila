import torch
import yaml

import wandb
from einops import rearrange

from models.unet import SensorDataUnet
import matplotlib.pyplot as plt

def apply_colormap(images, cmap='jet'):
    # Get the colormap
    cm = plt.get_cmap(cmap)

    colored_images = cm(images[:, :, :, 0])

    # Convert RGBA to RGB
    rgb_images = colored_images[:, :, :, :3]

    return rgb_images

def load_config(config_file_path):
    with open(config_file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_config = config["models"]
    optim_config = config["optimizer"]
    dataset_config = config["dataset"]
    test_dataset_config = config["test_dataset"]
    dataloader_config = config["dataloader"]
    test_dataloader_config = config["test_dataloader"]
    train_config = config["train"]
    image_size = train_config["image_size"]

    return config, model_config, optim_config, dataset_config, test_dataset_config, dataloader_config, test_dataloader_config, train_config, image_size


def eval_unet(unet, accelerator, test_dataloader, step, test_transform,
              max_num_images=5, color_map='jet'):
    unet.eval()
    with torch.no_grad():
        test_loss = 0
        test_outputs = []
        test_batches = []
        encodings = []
        for test_batch in test_dataloader:
            y = test_batch[0]
            y = test_transform(y)
            y = y.to(torch.float32).to(accelerator.device)
            y_pred, encoding = unet.forward_with_encoder(y)
            test_loss += torch.nn.functional.mse_loss(y_pred, y)

            if len(test_batches) < max_num_images:
                test_batches.append(y[0].unsqueeze(0))
                test_outputs.append(y_pred[0].unsqueeze(0))
                encodings.append(encoding[0].unsqueeze(0))
                if len(test_batches) >= max_num_images:
                    break

        test_loss /= len(test_dataloader)
        test_batch = torch.cat(test_batches, dim=0)
        test_output = torch.cat(test_outputs, dim=0)
        encoding = torch.cat(encodings, dim=0)

        test_batch = rearrange(test_batch, 'b c h w -> b h w c').detach().cpu().numpy()
        test_output = rearrange(test_output, 'b c h w -> b h w c').detach().cpu().numpy()
        encoding = rearrange(encoding, 'b c h w -> b h w c').detach().cpu().numpy()
        test_batch = apply_colormap(test_batch, cmap=color_map)
        test_output = apply_colormap(test_output, cmap=color_map)
        test_batch_images = [wandb.Image(img) for img in test_batch]
        test_output_images = [wandb.Image(img) for img in test_output]
        encoding_images = {}
        if encoding.shape[-1] not in [1, 3]:
            for c in range(encoding.shape[-1]):
                encoding_images[f"test encoding channel_{c}"] = [wandb.Image(img) for img in encoding[:, :, :, c]]
        else:
            encoding_images["test encoding"] = [wandb.Image(img) for img in encoding]

        accelerator.log({"test_loss": test_loss, "test_target": test_batch_images,
                         "test_output": test_output_images, **encoding_images},
                        step=step)
    unet.train()

def log_images(gt, y_pred, y, accelerator, step, color_map="jet"):
    gt = rearrange(gt, 'b c h w -> b h w c').detach().cpu().numpy()
    y = rearrange(y, 'b c h w -> b h w c').detach().cpu().numpy()
    y_pred = rearrange(y_pred, 'b c h w -> b h w c').detach().cpu().numpy()
    gt = apply_colormap(gt, cmap=color_map)
    y = apply_colormap(y, cmap=color_map)
    y_pred = apply_colormap(y_pred, cmap=color_map)
    gt = [wandb.Image(img) for img in gt]
    y_pred = [wandb.Image(img) for img in y_pred]
    y = [wandb.Image(img) for img in y]
    accelerator.log({"gt": gt, "y_pred": y_pred, "y": y}, step=step)

def eval_sensor_data_unet(data_unet: SensorDataUnet, accelerator, test_dataloader, step,
                          max_num_images=5, color_map="jet"):
    data_unet.eval()
    with torch.no_grad():
        test_loss = 0
        test_outputs = []
        test_batches = []
        encodings = []
        for test_batch in test_dataloader:
            y, x, _ = test_batch
            y = y.to(torch.float32).to(accelerator.device)
            x = x.to(torch.float32).to(accelerator.device)
            y_pred, encoding = data_unet.forward_with_encoder(x, y.shape[2:])
            test_loss += torch.nn.functional.mse_loss(y_pred, y)

            if len(test_batches) < max_num_images:
                for i in range(y.shape[0]):
                    test_batches.append(y[i].unsqueeze(0))
                    test_outputs.append(y_pred[i].unsqueeze(0))
                    encodings.append(encoding[i].unsqueeze(0))
                    if len(test_batches) >= max_num_images:
                        break

        test_loss /= len(test_dataloader)

        test_batch = torch.cat(test_batches, dim=0)
        test_output = torch.cat(test_outputs, dim=0)
        encoding = torch.cat(encodings, dim=0)

        test_batch = rearrange(test_batch, 'b c h w -> b h w c').detach().cpu().numpy()
        test_output = rearrange(test_output, 'b c h w -> b h w c').detach().cpu().numpy()
        test_batch = apply_colormap(test_batch, cmap=color_map)
        test_output = apply_colormap(test_output, cmap=color_map)
        encoding = rearrange(encoding, 'b c h w -> b h w c').detach().cpu().numpy()
        test_batch_images = [wandb.Image(img) for img in test_batch]
        test_output_images = [wandb.Image(img) for img in test_output]
        encoding_images = {}
        if encoding.shape[-1] not in [1, 3]:
            for c in range(encoding.shape[-1]):
                encoding_images[f"test encoding channel_{c}"] = [wandb.Image(img) for img in encoding[:, :, :, c]]
        else:
            encoding_images["test encoding"] = [wandb.Image(img) for img in encoding]

        accelerator.log({"test_loss": test_loss, "test_target": test_batch_images,
                         "test_output": test_output_images, **encoding_images},
                        step=step)
    data_unet.train()