import torch

def train_model(model,train_loader,loss_function,optimizer,nb_epochs=10,device='cpu'):
    for epoch in range(nb_epochs):
        print(f"Epoch {epoch+1}\n-------------------")
        model.train()
        running_loss = 0.0
        for batch, (X,y) in enumerate(train_loader):
            X,y = X.to(device), y.to(device) #pass the data to the CPU / GPU 

            #forward pass
            pred = model(X)  
            loss = loss_function(pred,y) #calculate the loss 

            #backpropagation
            optimizer.zero_grad() #reset gradients to 0 
            loss.backward() #backpropagation 
            optimizer.step() #update weights 

            running_loss += loss.item()
            if batch % 100 == 0: 
                print(f"Loss at batch {batch} : {loss.item()}") 
        
        print(f"Average loss at epoch {epoch+1} : {running_loss/len(train_loader)}")

    print("Training Complete") 

def evaluate_model(model,test_loader,device='cpu'):
    model.eval() 
    correct = 0 
    total = 0 
    with torch.no_grad():
        for X,y in test_loader:
            X,y = X.to(device), y.to(device) 
            pred = model(X)
            predicted = pred.argmax(1) #class with the highest probability
            correct += (predicted == y).sum().item() #number of good predictions 
            total += y.size(0)
    accuracy = correct / total 
    print(f"Accuracy : {accuracy:.2f}")