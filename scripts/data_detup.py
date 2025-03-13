import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def get_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int=NUM_WORKERS
):
    """Creates Training and Testing DataLoaders
    Takes in a training directory and testing directory and turns
    them into PyTorch Datasets and then into PyTorch Dataloaders

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: torhvision tranforms to perform on the training and test images
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
    
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
            test_dir=path/to/test_dir,
            transform=some_transform,
            batch_size=32,
            num_workers=4)
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(root=train_dir,transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # Get class names list from the dataset
    class_names = train_data.classes

    # Create the train and test dataloaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count()
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count()
    )

    # Return the dataloaders
    return train_dataloader, test_dataloader, class_names