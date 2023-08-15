import argparse

import torch
import torchinfo
import torchvision
import yaml
from accelerate import Accelerator
from tqdm import tqdm

from dataset import ObservationDataset, get_y_test_transforms
from models.unet import SensorDataUnet
from utils import log_images, eval_sensor_data_unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, default="config/base.yml")
    parser.add_argument("--unet_checkpoint", "-u", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_config = config["models"]
    optim_config = config["optimizer"]
    dataset_config = config["dataset"]
    test_dataset_config = config["test_dataset"]
    dataloader_config = config["dataloader"]
    test_dataloader_config = config["test_dataloader"]
    train_config = config["train"]
    image_size = train_config["image_size"]

    # Tell the Accelerator object to log with wandb
    accelerator = Accelerator(log_with="wandb")

    data_unet = SensorDataUnet(**model_config["data_unet"])
    data_unet.train()

    if args.unet_checkpoint is not None:
        torch.load(args.unet_checkpoint)
        data_unet.load_state_dict(torch.load(args.unet_checkpoint), strict=False)

        # freeze decoder
        for param in data_unet.decoder.parameters():
            param.requires_grad = False

    torchinfo.summary(data_unet, input_data={
        "x": torch.randn(1, 2, 16, 16),
        "output_size": (256, 256),
    })

    optimizer = torch.optim.AdamW(data_unet.parameters(), **optim_config["kwargs"])

    dataset = ObservationDataset(**dataset_config,
                                 y_transform=torchvision.transforms.Compose(
                                     [torchvision.transforms.ToTensor(), ]
                                 ),
                                 x_transform=torchvision.transforms.Compose(
                                     [
                                         torchvision.transforms.ToTensor(),
                                     ]
                                 ))

    test_dataset = ObservationDataset(**test_dataset_config,
                                      y_transform=torchvision.transforms.Compose(
                                          [torchvision.transforms.ToTensor(), ]
                                      ),
                                      x_transform=torchvision.transforms.Compose(
                                          [
                                              torchvision.transforms.ToTensor(),
                                          ]
                                      )
                                      )

    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_config)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_dataloader_config)

    data_unet, optimizer, dataloader = accelerator.prepare(data_unet, optimizer, dataloader)

    test_transform = get_y_test_transforms(image_size)

    # Initialise your wandb run, passing wandb parameters and any config information
    accelerator.init_trackers(
        project_name=config["name"],
        config=config,
    )

    step = 0
    for epoch in tqdm(range(train_config["epochs"])):
        loss_since_last_log = 0
        for i, batch in tqdm(enumerate(dataloader)):
            with accelerator.accumulate(data_unet):
                y, x, _ = batch
                y = y.to(torch.float32)  # workaround
                x = x.to(torch.float32)  # workaround
                y = test_transform(y)
                y_pred = data_unet(x, y.shape[2:])
                loss = torch.nn.functional.mse_loss(y_pred, y)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(data_unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                loss_since_last_log += loss

                if (epoch + 1) * i % train_config["log_interval"] == 0:
                    accelerator.log({"loss": loss_since_last_log / train_config["log_interval"]})
                    loss_since_last_log = 0

                if (epoch + 1) * i % train_config["log_interval"] == 0:
                    log_images(y, y_pred, y, accelerator, step)
                if (epoch + 1) * i % train_config["eval_interval"] == 0:
                    eval_sensor_data_unet(data_unet, accelerator, test_dataloader, step)
            step += 1

        if (epoch + 1) % train_config["save_interval"] == 0:
            accelerator.save(data_unet.state_dict(), f"data_unet_{epoch}.pth")

    accelerator.end_training()
