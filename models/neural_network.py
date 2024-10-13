import torch
from torch import nn


class NeuralNetwork(nn.Module): #create a model object containing the layers and structure of a neuronal network
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #create a "flattening" layer => input data are flattened from 28*28 to 784 elements
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512), #first linear layer take a 784 elts vector and produce 512 outputs elts
            nn.ReLU(), #apply ReLU function => x >= 0 = x & x < 0 = 0
            nn.Linear(512,512), #second linear layer take the first 512 elts and output 512 elts
            nn.ReLU(), #another ReLU function
            nn.Linear(512,10), #third layer -> take 512 elts and returns 10 elements (10th first numbers)
        ) 

    def forward(self,x): #define the propagation of the data through the network x = data
        x = self.flatten(x) #flatten the input data  
        logits = self.linear_relu_stack(x) #pass data through the differents layers 
        return logits #logits is a vector containing the scores for each class (from 0 to 9)
