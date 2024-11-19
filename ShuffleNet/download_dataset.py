import os
from torchvision import datasets, transforms
from collections import defaultdict
import torch
import torch.utils.data

class TargetTransform:
    def __init__(self, class_mapping, classes):
        self.class_mapping = class_mapping
        self.classes = classes

    def __call__(self, target):
        return self.class_mapping[self.classes[target]]

def get_data_loaders(data_dir, batch_size=2048, num_workers=16, pin_memory=True):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_datasets_list = []
    val_datasets_list = []

    # Step 1: Gather all class names and create a consistent mapping
    class_mapping = defaultdict(int)
    current_class = 0

    # Traverse through the subdirectories (trainx1, trainx2, ...) and collect class names
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            # Check if it's 'train' or 'val' folder and iterate through the classes
            if 'train' in subdir or 'val' in subdir:
                for class_name in os.listdir(subdir_path):
                    class_folder = os.path.join(subdir_path, class_name)
                    if os.path.isdir(class_folder):
                        # Print class names to see what is detected
                        print(f"Found class: {class_name} in subdir: {subdir}")
                        if class_name not in class_mapping:
                            class_mapping[class_name] = current_class
                            print(f"Assigned index {current_class} to class {class_name}")
                            current_class += 1

    print(f"Total unique classes found: {len(class_mapping)}")  # Should print 100
    print(f"Class mapping: {class_mapping}")  # Check the mapping

    # Step 2: Create datasets with the updated class_to_idx
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            if "train" in subdir:
                dataset = datasets.ImageFolder(root=subdir_path, transform=transform)
                target_transform = TargetTransform(class_mapping,dataset.classes)
                dataset.target_transform = target_transform
                dataset.class_to_idx = {class_name: class_mapping[class_name] for class_name in dataset.classes}
                train_datasets_list.append(dataset)

            elif "val" in subdir:
                dataset = datasets.ImageFolder(root=subdir_path, transform=transform)
                target_transform = TargetTransform(class_mapping,dataset.classes)
                dataset.target_transform = target_transform
                dataset.class_to_idx = {class_name: class_mapping[class_name] for class_name in dataset.classes}
                val_datasets_list.append(dataset)

    # Step 3: Create the DataLoader for training and validation datasets
    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(train_datasets_list),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(val_datasets_list),
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader