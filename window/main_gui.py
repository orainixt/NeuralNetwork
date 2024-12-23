import tkinter as tk 
import torch
from models.neural_network import NeuralNetwork
from data.data_loader import load_data
from training.trainer import train_model, evaluate_model
import os 

class MainGui: 
    
    def __init__(self,root) :
        self.root = root 
        self.root.title("A.I") 
        self.canvas_width = 1280 
        self.canvas_height = 720 
        self.canvas = tk.Canvas(root,width=self.canvas_width,height=self.canvas_height,bg ="grey")
        self.canvas.pack()  

        self.device = ( #check which device to use
            "cuda" #if a nvidia gpu is avaible -> cuda
            if torch.cuda.is_available() 
            else "mps" #if an apple gpu is avaible -> Metal Perfomance Shaders
            if torch.backends.mps.is_available()
            else "cpu" #else using a generic cpu
        )

        self.label = tk.Label(root,text="A.I",)
        self.label.pack(pady=10)

        self.train_button = tk.Button(root,text="Train Model",command = train_model)
        self.evaluate_button = tk.Button(root,text="Evaluate Model",command = evaluate_model)

        self.train_button.pack(pady = 5)
        self.evaluate_button.pack(pady=5) 

        self.result_label = tk.Label(root,text="",fg="blue",wraplength=500, justify="left")
        self.result_label.pack(pady=20) 

        self.model = None 
        self.train_loader = None 
        self.test_loader = None 

    
    def load_data_and_model(self):
        """
        Function used to load the data and the model if it hasn't been initialized before 
        """
        if not self.train_loader or not self.test_loader:
            self.train_loader, self.test_loader = load_data()

        if not self.model:
            self.model = NeuralNetwork().to(self.device)

    def train_model(self):
        self.load_data_and_model()
        
        #initialize loss function and optimizer 
        loss_function = torch.nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        train_model(self.model, self.train_loader, loss_function, optimizer, nb_epochs=10, device=self.device)

        self.result_label.config(text="Training Complete :) !")

    def evaluate_model(self): 
        self.load_data_and_model()
        accuracy = evaluate_model(self.model,self.test_loader,device=self.device)
        self.result_label.config(text=f"Accuracy : {accuracy:.2f}")

