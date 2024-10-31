import torch
import torchvision

def get_data_loaders(batch_size_train=64, batch_size_test=64, download=True, num_workers=4):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('/files/', train=True, download=download,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.5,), (0.5,))
                                     ])),
        batch_size=batch_size_train, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('/files/', train=False, download=download,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.5,), (0.5,))
                                     ])),
        batch_size=batch_size_test, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader