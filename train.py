import torch
import torchvision
from einops import rearrange

import wandb
import yaml
from accelerate import Accelerator
import argparse
import torchinfo
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from dataset import ObservationDataset, AddGaussianNoise, target_transform
from models.unet import Unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, default="config/base.yml")

    return parser.parse_args()


args = parse_args()

with open(args.config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_config = config["models"]
optim_config = config["optimizer"]
dataset_config = config["dataset"]
dataloader_config = config["dataloader"]
train_config = config["train"]
image_size = train_config["image_size"]

# Tell the Accelerator object to log with wandb
accelerator = Accelerator(log_with="wandb")

# Initialise your wandb run, passing wandb parameters and any config information
accelerator.init_trackers(
    project_name=config["name"],
    config=config,
)

unet = Unet(**model_config["unet"])

torchinfo.summary(unet, input_size=(1, 1, image_size, image_size))

encoder_output = unet.encoder(torch.randn(1, 1, 256, 256, device=accelerator.device), )
print(f"Encoder output shape: {encoder_output.shape}")

optimizer = torch.optim.AdamW(unet.parameters(), **optim_config["kwargs"])

dataset = ObservationDataset(**dataset_config,
                             y_transform=torchvision.transforms.Compose(
                                 [torchvision.transforms.ToTensor(), ]
                             ),
                             )

test_dataset = ObservationDataset(**dataset_config,
                                  y_transform=torchvision.transforms.Compose(
                                      [torchvision.transforms.ToTensor(), ]
                                  ),
                                  )

dataloader = torch.utils.data.DataLoader(dataset, **dataloader_config)

unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

for epoch in tqdm(range(train_config["epochs"])):
    for i, batch in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(unet):
            batch = batch[0]
            batch = batch.to(torch.float32)  # workaround
            input_ = target_transform(batch)
            output = unet(input_)
            loss = torch.nn.functional.mse_loss(output, input_)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(unet.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

            if (epoch + 1) * i % train_config["log_interval"] == 0:
                accelerator.log({"loss": loss})

            if (epoch + 1) * i % train_config["log_interval"] == 0:
                batch = rearrange(batch, 'b c h w -> b h w c').detach().cpu().numpy()
                output = rearrange(output, 'b c h w -> b h w c').detach().cpu().numpy()
                input_ = rearrange(input_, 'b c h w -> b h w c').detach().cpu().numpy()
                in_images = [wandb.Image(img) for img in batch]
                out_images = [wandb.Image(img) for img in output]
                input_images = [wandb.Image(img) for img in input_]
                accelerator.log({"target": in_images, "output": out_images,
                                 "input": input_images})
            if (epoch + 1) * i % train_config["eval_interval"] == 0:
                unet.eval()
                with torch.no_grad():
                    test_batch = next(iter(test_dataset))
                    test_batch = test_batch[0].unsqueeze(0)
                    test_batch = test_batch.to(torch.float32).to(accelerator.device)
                    test_output = unet(test_batch)
                    test_loss = torch.nn.functional.mse_loss(test_output, test_batch)
                    test_batch = rearrange(test_batch, 'b c h w -> b h w c').detach().cpu().numpy()
                    test_output = rearrange(test_output, 'b c h w -> b h w c').detach().cpu().numpy()
                    test_batch_images = [wandb.Image(img) for img in test_batch]
                    test_output_images = [wandb.Image(img) for img in test_output]
                    accelerator.log({"test_loss": test_loss, "test_target": test_batch_images,
                                     "test_output": test_output_images}, step=(epoch + 1) * i)


    accelerator.save(unet.state_dict(), f"unet_{epoch}.pth")

accelerator.end_training()
