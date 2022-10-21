import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import config.resnet_config as resnet_config
import backbone.resnet as resnet
from data.dataset import CUDAPrefetcher, ImageDataset
from utils.torch_utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter

model_names = sorted(
    name for name in resnet.__dict__ if name.islower() and not name.startswith("__") and callable(resnet.__dict__[name]))


def build_model() -> nn.Module:
    resnet_model = resnet.__dict__[resnet_config.model_arch_name](num_classes=resnet_config.model_num_classes)
    resnet_model = resnet_model.to(device=resnet_config.device, memory_format=torch.channels_last)

    return resnet_model


def load_dataset() -> CUDAPrefetcher:
    test_dataset = ImageDataset(resnet_config.test_image_dir,
                                resnet_config.image_size,
                                resnet_config.model_mean_parameters,
                                resnet_config.model_std_parameters,
                                "Test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=resnet_config.batch_size,
                                 shuffle=False,
                                 num_workers=resnet_config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    test_prefetcher = CUDAPrefetcher(test_dataloader, resnet_config.device)

    return test_prefetcher


def main() -> None:
    # Initialize the model
    resnet_model = build_model()
    print(f"Build `{resnet_config.model_arch_name}` model successfully.")

    # Load model weights
    resnet_model, _, _, _, _, _ = load_state_dict(resnet_model, resnet_config.model_weights_path)
    print(f"Load `{resnet_config.model_arch_name}` "
          f"model weights `{os.path.abspath(resnet_config.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    resnet_model.eval()

    # Load test dataloader
    test_prefetcher = load_dataset()

    # Calculate how many batches of data are in each Epoch
    batches = len(test_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"Test: ")

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=resnet_config.device, non_blocking=True)
            target = batch_data["target"].to(device=resnet_config.device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Inference
            output = resnet_model(images)

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0].item(), batch_size)
            acc5.update(top5[0].item(), batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % resnet_config.test_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = test_prefetcher.next()

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    print(f"Acc@1 error: {100 - acc1.avg:.2f}%")
    print(f"Acc@5 error: {100 - acc5.avg:.2f}%")


if __name__ == "__main__":
    main()
