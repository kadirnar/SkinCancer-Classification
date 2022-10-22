import torchvision
import matplotlib.pyplot as plt
import numpy as np

def multi_imshow(data_loader):
    images, labels = next(iter(data_loader))
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def read_yaml(yaml_file):
    """
    Read yaml file
    Args:
        file_path: str
    """
    import yaml

    with open(yaml_file, "r") as stream:
        data = yaml.safe_load(stream)

    return data