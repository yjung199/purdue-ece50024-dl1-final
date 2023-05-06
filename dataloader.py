from __future__ import division, print_function, absolute_import

import os
import glob
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as PILI
import numpy as np


class ClassDataset(Dataset):
    def __init__(self, images: List[str], label: int, transform=None):
        self.images = images
        self.label = label
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = PILI.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, self.label

    def __len__(self) -> int:
        return len(self.images)


class EpisodeDataset(Dataset):
    def __init__(self, root: str, mode: str = 'train', num_shot: int = 5, num_eval: int = 15, transform=None):
        self.root = os.path.join(root, mode)
        self.labels = sorted(os.listdir(self.root))
        img_w_labels = [glob.glob(os.path.join(root, label, '*')) for label in self.labels]
        self.episode_loader = [DataLoader(ClassDataset(glob.glob(os.path.join(self.root, label, '*')), idx, transform), batch_size=num_shot+num_eval, shuffle=True, num_workers=0) for idx, label in enumerate(self.labels)]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(iter(self.episode_loader[idx]))

    def __len__(self) -> int:
        return len(self.labels)

class EpisodicSampler(Sampler):
    def __init__(self, total_classes: int, num_class: int, num_episode: int):
        self.total_classes = total_classes
        self.num_class = num_class
        self.num_episode = num_episode

    def __iter__(self):
        for _ in range(self.num_episode):
            yield torch.randperm(self.total_classes)[:self.num_class]

    def __len__(self) -> int:
        return self.num_episode

def prepare_data(args):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
        )
    
    jitter = transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            )
    
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            jitter,
            transforms.ToTensor(),
            normalize
        ])

    transform_val = transforms.Compose([
            # transforms.Resize(args.image_size * 1.15),
            transforms.Resize(args.image_size * 8//7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize])
    
    transform_test = transforms.Compose([
            # transforms.Resize(args.image_size * 1.15),
            transforms.Resize(args.image_size * 8//7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize])
    
    train_set = EpisodeDataset(
        root=args.data_root,
        mode='train',
        num_shot=args.num_shot,
        num_eval=args.num_eval,
        transform=transform_train
    )

    val_set = EpisodeDataset(
        root=args.data_root,
        mode='val',
        num_shot=args.num_shot,
        num_eval=args.num_eval,
        transform=transform_val
    )

    test_set = EpisodeDataset(
        root=args.data_root,
        mode='test',
        num_shot=args.num_shot,
        num_eval=args.num_eval,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        batch_sampler=EpisodicSampler(len(train_set), args.num_class, args.episode)
    )

    val_loader = DataLoader(
        val_set,
        num_workers=2,
        pin_memory=False,
        batch_sampler=EpisodicSampler(len(val_set), args.num_class, args.episode_val)
    )

    test_loader = DataLoader(
        test_set,
        num_workers=2,
        pin_memory=False,
        batch_sampler=EpisodicSampler(len(test_set), args.num_class, args.episode_val)
    )

    return train_loader, val_loader, test_loader