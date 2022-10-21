import argparse
import json
import os

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import Resize, ConvertImageDtype, Normalize

import utils.imgproc as imgproc
import backbone.resnet as resnet
from utils.torch_utils import load_state_dict

model_names = sorted(
    name for name in resnet.__dict__ if name.islower() and not name.startswith("__") and callable(resnet.__dict__[name]))


def load_class_label(class_label_file: str, num_classes: int) -> list:
    class_label = json.load(open(class_label_file))
    class_label_list = [class_label[str(i)] for i in range(num_classes)]

    return class_label_list


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, model_num_classes: int, device: torch.device) -> [nn.Module, nn.Module]:
    resnet_model = resnet.__dict__[model_arch_name](num_classes=model_num_classes)
    resnet_model = resnet_model.to(device=device, memory_format=torch.channels_last)

    return resnet_model


def preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    image = cv2.imread(image_path)

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # OpenCV convert PIL
    image = Image.fromarray(image)

    # Resize to 224
    image = Resize([image_size, image_size])(image)
    # Convert image data to pytorch format data
    tensor = imgproc.image_to_tensor(image, False, False).unsqueeze_(0)
    # Convert a tensor image to the given ``dtype`` and scale the values accordingly
    tensor = ConvertImageDtype(torch.float)(tensor)
    # Normalize a tensor image with mean and standard deviation.
    tensor = Normalize(args.model_mean_parameters, args.model_std_parameters)(tensor)

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

    return tensor


def main():
    # Get the label name corresponding to the drawing
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes)

    device = choice_device(args.device_type)

    # Initialize the model
    resnet_model = build_model(args.model_arch_name, args.model_num_classes, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    resnet_model, _, _, _, _, _ = load_state_dict(resnet_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    resnet_model.eval()

    tensor = preprocess_image(args.image_path, args.image_size, device)

    # Inference
    with torch.no_grad():
        output = resnet_model(tensor)

    # Calculate the five categories with the highest classification probability
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist()

    # Print classification results
    for class_index in prediction_class_index:
        prediction_class_label = class_label_map[class_index]
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch_name", type=str, default="resnet18")
    parser.add_argument("--model_mean_parameters", type=list, default=[0.485, 0.456, 0.406])
    parser.add_argument("--model_std_parameters", type=list, default=[0.229, 0.224, 0.225])
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")
    parser.add_argument("--model_num_classes", type=int, default=1000)
    parser.add_argument("--model_weights_path", type=str, default="./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar")
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device_type", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    main()
