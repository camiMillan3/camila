import torch
import torchvision

import wandb
from accelerate import Accelerator
import argparse
import torchinfo
from tqdm import tqdm

from dataset import ObservationDataset, get_y_train_transforms, get_y_test_transforms
from models.unet import Unet
from utils import eval_unet, log_images, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, default="config/base.yml")
    parser.add_argument("--unet_checkpoint", "-u", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    (config, model_config, optim_config, dataset_config, test_dataset_config,
     dataloader_config, test_dataloader_config, train_config, image_size) = load_config(args.config_file)
    # Tell the Accelerator object to log with wandb
    accelerator = Accelerator(log_with="wandb")

    print("Using device:", accelerator.device)

    # Initialise your wandb run, passing wandb parameters and any config information
    accelerator.init_trackers(
        project_name=config["name"] + "_unet",
        config=config,
    )

    unet = Unet(**model_config["unet"])
    unet.train()

    wandb.watch(unet, log="all", log_freq=300)

    if args.unet_checkpoint is not None:
        torch.load(args.unet_checkpoint)
        unet.load_state_dict(torch.load(args.unet_checkpoint), strict=True)
        print(f"Loaded checkpoint from {args.unet_checkpoint}")

    torchinfo.summary(unet, input_size=(1, 1, image_size, image_size))

    encoder_output = unet.encoder(torch.randn(1, 1, 256, 256, device=accelerator.device), )
    print(f"Encoder output shape: {encoder_output.shape}")

    optimizer = torch.optim.AdamW(unet.parameters(), **optim_config["kwargs"])

    dataset = ObservationDataset(**dataset_config,
                                 y_transform=torchvision.transforms.Compose(
                                     [torchvision.transforms.ToTensor(), ]
                                 ),
                                 )

    test_dataset = ObservationDataset(**test_dataset_config,
                                      y_transform=torchvision.transforms.Compose(
                                          [torchvision.transforms.ToTensor(), ]
                                      ),
                                      )

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_config)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_dataloader_config)

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    train_transform = get_y_train_transforms(image_size)
    test_transform = get_y_test_transforms(image_size)

    step = 0
    for epoch in tqdm(range(train_config["epochs"])):
        loss_since_last_log = 0

        for i, batch in tqdm(enumerate(dataloader)):
            with accelerator.accumulate(unet):
                batch = batch[0]
                gt = batch.to(torch.float32)  # workaround
                y = train_transform(gt)
                y_pred = unet(y)
                loss = torch.nn.functional.mse_loss(y_pred, y)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                loss_since_last_log += loss.item()

                if step % train_config["log_interval"] == 0:
                    accelerator.log({"loss": loss_since_last_log / train_config["log_interval"]}, step=step)
                    loss_since_last_log = 0

                if step % train_config["log_interval"] == 0:
                    log_images(gt, y_pred, y, accelerator, step)
                if (epoch + 1) * i % train_config["eval_interval"] == 0:
                    eval_unet(unet, accelerator, test_dataloader, step=step,
                              test_transform=test_transform)
            step += 1

        if (epoch + 1) % train_config["save_interval"] == 0:
            accelerator.save(unet.state_dict(), f"unet_{epoch}.pth")

    accelerator.end_training()
