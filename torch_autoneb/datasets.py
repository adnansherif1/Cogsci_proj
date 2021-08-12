from itertools import product, repeat
import torch
from os.path import join, dirname
from torch import FloatTensor, LongTensor, normal
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
from torchvision.transforms import Compose, Pad, ToTensor, RandomCrop, RandomHorizontalFlip

class RandDataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inp = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            inp = self.transform(self.data[idx])
        if self.target_transform:
            label = self.target_transform(self.labels[idx])
        return inp, label

def load_dataset(name):
    root = join(dirname(dirname(__file__)), "tmp", name)

    if "cifar" in name:
        if name == "cifar10":
            cls = CIFAR10
            class_count = 10
        elif name == "cifar100":
            cls = CIFAR100
            class_count = 100
        else:
            raise ValueError(f"Unknown CIFAR dataset {name}")

        return {
            "train": cls(root, train=True, transform=Compose([RandomCrop(32, 4), RandomHorizontalFlip(), ToTensor()]), download=True),
            "test": cls(root, train=False, transform=ToTensor(), download=True),
        }, (3, 32, 32), class_count
    elif name == "mnist":
        # Make size power of two for architecture convenience
        transform = Compose([Pad(2), ToTensor()])
        return {
            "train": MNIST(root, train=True, transform=transform, download=True),
            "test": MNIST(root, train=False, transform=transform, download=True),
        }, (32, 32), 10
    elif name == "xor":
        pass
    elif name == "custom":
        data = [torch.rand(1,2) for i in range(10)]
        labels = torch.randint(0, 10, (10,))

        return {
            "train": RandDataset(data,labels),
            "test": RandDataset(data,labels),
        }, (2,), 10
    raise ValueError(f"Unknown dataset {name}")


class XORDataset(Dataset):
    def __init__(self, train, transform=None, target_transform=None):
        self.train = train

        if train:
            size = 500
        else:
            size = 100
        each_size = size // 4
        assert each_size * 4 == size

        self.data = FloatTensor(size, 2)
        self.target = LongTensor(size)
        offset = 0
        for x, y in product((-1, 1), (-1, 1)):
            self.data[offset:offset + each_size] = normal(FloatTensor(list(repeat((x, y), each_size))), 0.2)
            self.target[offset:offset + each_size] = 1 if x == y else 0
            offset += each_size

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        target = self.target[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
