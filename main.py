import torch
from models.neural_network import NeuralNetwork
from data.data_loader import load_data
from training.trainer import train_model, evaluate_model

device = ( #check which device to use
    "cuda" #if a nvidia gpu is avaible -> cuda
    if torch.cuda.is_available() 
    else "mps" #if an apple gpu is avaible -> Metal Perfomance Shaders
    if torch.backends.mps.is_available()
    else "cpu" #else using a generic cpu
)

#load data 
train_loader, test_loader = load_data()

#instanciate model, loss function and Adam optimizer
model = NeuralNetwork().to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, loss_function, optimizer, num_epochs=10, device=device)

evaluate_model(model, test_loader, device=device)