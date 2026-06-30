import os, sys
import random
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from torch import nn
import numpy as np
import time
from functools import wraps
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def note_print(*args, **kwargs):
    """Print in green for visibility in the terminal."""
    print("\033[0;32m", *args, "\033[0m", **kwargs)


def get_dynamic_clip_norm(epoch, total_epochs, start=1.0, end=0.01, decay_every=5):
    if epoch % decay_every != 0:
        return None
    ratio = epoch / total_epochs
    return start * (end / start) ** ratio


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Time taken by {func.__name__}: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f} (hh:mm:ss)")
        return result
    return wrapper


def seed_torch(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class Identity:
    def __call__(self, x):
        return x


# Per-dataset channel mean/std used for normalization.
DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def get_transforms(dataset_name, model_name, wo_dataaug):
    """CIFAR-10/100 transforms. resnet18 keeps 32x32; vgg16 uses Identity (no resize)."""
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset_name} (cifar10 or cifar100)")
    mean, std = DATASET_STATS[dataset_name]

    resize_transform = Identity() if model_name == "vgg16" else transforms.Resize((32, 32))
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        resize_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        resize_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if wo_dataaug:
        transform_train = transform_test
    return transform_train, transform_test


def get_dataset(dataset_name, transform_train, transform_test, path=Path("./data").expanduser()):
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root=path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name} (cifar10 or cifar100)")
    return train_dataset, test_dataset


def get_dataloader(trainset, testset, batch_size, num_workers, shuffle=True):
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def split_class_data(dataset, forget_class_index, num_forget):
    """
    Split a dataset into forget / remain index lists.

    Returns:
        forget_index: sample indices of the forget classes (capped at num_forget)
        remain_index: all non-forget samples + any forget samples beyond num_forget
        class_remain_index: forget-class samples not included in forget_index
    """
    targets = dataset.targets
    targets = torch.tensor(targets)

    forget_class_index = torch.tensor(forget_class_index)
    mask = torch.isin(targets, forget_class_index)
    forget_class_indices = torch.nonzero(mask).flatten()

    assert forget_class_indices.numel() > 0, f"No samples found for class in {forget_class_indices}"

    num_forget = min(num_forget, forget_class_indices.numel())
    forget_index = forget_class_indices[:num_forget]
    class_remain_index = forget_class_indices[num_forget:]

    remain_index = torch.nonzero(~mask).flatten()
    remain_index = torch.cat((remain_index, class_remain_index))

    return forget_index.tolist(), remain_index.tolist(), class_remain_index.tolist()


def get_unlearn_loader(trainset, testset, forget_class_index, batch_size, num_forget, num_workers, repair_num_ratio=0.01):
    """
    Pull the requested number of forget-class samples out of the train set, and all
    forget-class samples out of the test set, building forget/remain loaders for both.
    """
    train_forget_index, train_remain_index, class_remain_index = split_class_data(trainset, forget_class_index, num_forget=num_forget)
    assert isinstance(train_forget_index, list)
    test_forget_index, test_remain_index, _ = split_class_data(testset, forget_class_index, num_forget=len(testset))

    repair_class_index = random.sample(class_remain_index, int(repair_num_ratio * len(class_remain_index)))

    train_forget_sampler = SubsetRandomSampler(train_forget_index)
    train_remain_sampler = SubsetRandomSampler(train_remain_index)
    repair_class_sampler = SubsetRandomSampler(repair_class_index)
    test_forget_sampler = SubsetRandomSampler(test_forget_index)
    test_remain_sampler = SubsetRandomSampler(test_remain_index)

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_forget_sampler, num_workers=num_workers)
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_remain_sampler, num_workers=num_workers)
    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=repair_class_sampler, num_workers=num_workers)
    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_forget_sampler, num_workers=num_workers)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_remain_sampler, num_workers=num_workers)

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index
