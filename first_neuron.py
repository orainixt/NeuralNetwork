import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda" #if a nvidia gpu is avaible -> cuda
    if torch.cuda.is_available() 
    else "mps" #if an apple gpu is avaible -> Metal Perfomance Shaders
    if torch.backends.mps.is_available()
    else "cpu" #else using a generic cpu
)

print(f"Using {device} device")

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


model = NeuralNetwork().to(device) #create an instance of neuralnetwork and pass it to the good cpu
print(model)

x = torch.rand(1,28,28,device=device) #create a size tensor (1,28,28) filled with random values between 0 and 1
logits = model(x) #pass the tensor throught the network. The output is 10 raw predicted values for each class 
pred_probability = nn.Softmax(dim=1)(logits) #returns the prediction probabilities by passing logits through an instance of an nn.Softmax module
y_pred = pred_probability.argmax(1) #prediction probabilities
print(f"Predicted class : {y_pred}") 

input_image = torch.rand(3,28,28) #take 3 images of size 28*28 and spread it through the network 
#print(input_image.size()) #used to understand

flatten = nn.Flatten()
flat_image = flatten(input_image) #flatten the image to a contiguous array of 784 px values
#print(flat_image.size())

first_layer = nn.Linear(in_features=28*28,out_features=20) #linear transformation on the input
hidden_layer = first_layer(flat_image) #with stored bias & weights 
#print(hidden_layer.size())

#print(f"Before ReLU function {hidden_layer}\n\n")
hidden_layer = nn.ReLU()(hidden_layer) #ReLU function 
#print(f"After ReLU function {hidden_layer}")

sequential_modules = nn.Sequential( #the data is spread as it's defined in sequential_modules
    flatten,
    first_layer,
    nn.ReLU,
    nn.Linear(20,10)
)

input_image = torch.rand(3,28,28)
logits = (input_image) #logits - raw values in [-infinity, infinity] 

softmax = nn.Softmax(dim=1) #nn.Softmax module - dim parameter indicates the dimension along which the values must sum to 1.
pred_probability = softmax(logits) #


#layers are parameterized by weights and bias optimized during training process 
#nn.Module tracks all those fields. Each parameter print its size and a preview of its values

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")