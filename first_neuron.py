import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# device = (
#     "cuda" #if a nvidia gpu is avaible -> cuda
#     if torch.cuda.is_available() 
#     else "mps" #if an apple gpu is avaible -> Metal Perfomance Shaders
#     if torch.backends.mps.is_available()
#     else "cpu" #else using a generic cpu
# )

print(f"Using {device} device")

# class NeuralNetwork(nn.Module): #create a model object containing the layers and structure of a neuronal network
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten() #create a "flattening" layer => input data are flattened from 28*28 to 784 elements
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28,512), #first linear layer take a 784 elts vector and produce 512 outputs elts
#             nn.ReLU(), #apply ReLU function => x >= 0 = x & x < 0 = 0
#             nn.Linear(512,512), #second linear layer take the first 512 elts and output 512 elts
#             nn.ReLU(), #another ReLU function
#             nn.Linear(512,10), #third layer -> take 512 elts and returns 10 elements (10th first numbers)
#         ) 

#     def forward(self,x): #define the propagation of the data through the network x = data
#         x = self.flatten(x) #flatten the input data  
#         logits = self.linear_relu_stack(x) #pass data through the differents layers 
#         return logits #logits is a vector containing the scores for each class (from 0 to 9)


model = NeuralNetwork().to(device) #create an instance of neuralnetwork and pass it to the good cpu
print(model)

X = torch.rand(1,28,28,device=device) #create a size tensor (1,28,28) filled with random values between 0 and 1
logits = model(X) #pass the tensor throught the network. The output is 10 raw predicted values for each class 
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


# training_data = datasets.FashionMNIST(
#     root='data', #path where the train/test data is stored
#     train=True,  #true ? test : train 
#     download=True, #dwd from internet if not avalible in root
#     transform=ToTensor(),    
# )

# test_data = datasets.FashionMNIST(
#     root='data',
#     train=False,
#     download=True,
#     transform=ToTensor()
# )

# train_loader = DataLoader(training_data,batch_size=64,shuffle=True)
# test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

loss_function = nn.CrossEntropyLoss() #using CrossEntropy for classification 
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

nb_epochs = 10 
# for epoch in range(nb_epochs):
#     print(f"Epoch {epoch+1}\n-------------------")
#     model.train()
#     running_loss = 0.0
#     for batch, (X,y) in enumerate(train_loader):
#         X,y = X.to(device), y.to(device) #pass the data to the CPU / GPU 

#         #forward pass
#         pred = model(X)  
#         loss = loss_function(pred,y) #calculate the loss 

#         #backpropagation
#         optimizer.zero_grad() #reset gradients to 0 
#         loss.backward() #backpropagation 
#         optimizer.step() #update weights 

#         running_loss += loss.item()
#         if batch % 100 == 0: 
#             print(f"Loss at batch {batch} : {loss.item()}") 
        
#     print(f"Average loss at epoch {epoch+1} : {running_loss/len(train_loader)}")

# print("Training Complete")

#evaluation mode 
# model.eval() 
# correct = 0 
# total = 0 
# with torch.no_grad():
#     for X,y in test_loader:
#         X,y = X.to(device), y.to(device) 
#         pred = model(X)
#         predicted = pred.argmax(1) #class with the highest probability
#         correct += (predicted == y).sum().item() #number of good predictions 
#         total += y.size(0)


