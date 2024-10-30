import torch 
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 

def predict_digit(model,image_path,device='cpu'):

    image = Image.open(image_path).convert('L') #convert('L') => convert in B&W
    image = image.resize((28,28)) #resize to 28 by 28 pixels 
    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    image = image.to(device)

    model.eval() 
    
    with torch.no_grad(): 
        output = model(image)
        prediction = output.argmax(1).item()
        plt.imshow(image.squeeze().cpu().numpy(), cmap="gray") 
        plt.title(f"Prediction: {prediction}")
        plt.show()

    return prediction  
