import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def load_data(batch_size=64):
    training_data = datasets.FashionMNIST(
    root='data', #path where the train/test data is stored
    train=True,  #true ? test : train 
    download=True, #dwd from internet if not avalible in root
    transform=ToTensor(),    
    )

    test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
    )
    
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
