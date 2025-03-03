import os
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

def get_dataloader(cid, batch_size: int, workers: int, ty: str):
    """Generates train, val, and test dataloaders."""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),          
    ])

    dataset = datasets.ImageFolder("/Users/iyadwehbe/Desktop/dataset/" +str (cid), transform=transform)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size  

    
    train_dataset, test_dataset = random_split(dataset, [train_size,  test_size])

    
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    if ty == 'train':
        dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    elif ty == 'test':
        dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError(f"Invalid type '{ty}' passed. Expected one of 'train', or 'test'.")
    
    return dloader
