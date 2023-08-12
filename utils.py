import torch
import wandb
from einops import rearrange

from models.unet import SensorDataUnet


def eval_unet(unet, accelerator, test_dataloader, step, test_transform):
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
            test_batches.append(y)
            test_outputs.append(y_pred)
            encodings.append(encoding)

        test_loss /= len(test_dataloader)
        test_batch = torch.cat(test_batches, dim=0)
        test_output = torch.cat(test_outputs, dim=0)
        encoding = torch.cat(encodings, dim=0)

        test_batch = rearrange(test_batch, 'b c h w -> b h w c').detach().cpu().numpy()
        test_output = rearrange(test_output, 'b c h w -> b h w c').detach().cpu().numpy()
        encoding = rearrange(encoding, 'b c h w -> b h w c').detach().cpu().numpy()
        test_batch_images = [wandb.Image(img) for img in test_batch]
        test_output_images = [wandb.Image(img) for img in test_output]
        encoding_images = [wandb.Image(img) for img in encoding]

        accelerator.log({"test_loss": test_loss, "test_target": test_batch_images,
                         "test_output": test_output_images, "encoding": encoding_images},
                        step=step)
    unet.train()

def log_images(gt, y_pred, y, accelerator, step):
    gt = rearrange(gt, 'b c h w -> b h w c').detach().cpu().numpy()
    y = rearrange(y, 'b c h w -> b h w c').detach().cpu().numpy()
    y_pred = rearrange(y_pred, 'b c h w -> b h w c').detach().cpu().numpy()
    gt = [wandb.Image(img) for img in gt]
    y_pred = [wandb.Image(img) for img in y_pred]
    y = [wandb.Image(img) for img in y]
    accelerator.log({"gt": gt, "y_pred": y_pred, "y": y}, step=step)

def eval_sensor_data_unet(data_unet: SensorDataUnet, accelerator, test_dataloader, step):
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
            test_batches.append(y)
            test_outputs.append(y_pred)
            encodings.append(encoding)

        test_loss /= len(test_dataloader)
        test_batch = torch.cat(test_batches, dim=0)
        test_output = torch.cat(test_outputs, dim=0)
        encoding = torch.cat(encodings, dim=0)

        test_batch = rearrange(test_batch, 'b c h w -> b h w c').detach().cpu().numpy()
        test_output = rearrange(test_output, 'b c h w -> b h w c').detach().cpu().numpy()
        encoding = rearrange(encoding, 'b c h w -> b h w c').detach().cpu().numpy()
        test_batch_images = [wandb.Image(img) for img in test_batch]
        test_output_images = [wandb.Image(img) for img in test_output]
        encoding_images = [wandb.Image(img) for img in encoding]

        accelerator.log({"test_loss": test_loss, "test_target": test_batch_images,
                         "test_output": test_output_images, "encoding": encoding_images},
                        step=step)
    data_unet.train()