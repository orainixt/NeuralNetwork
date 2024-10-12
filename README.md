# This is a project where I try to understand how a basic neural network is created 

# Introduction : 
- First of all, I randomly saw those videos on youtube : https://www.youtube.com/watch?v=DQ0lCm0J3PM / https://youtu.be/cAkMcPfY_Ns?si=iUUTHvFgFkVngtIy
- The creator explained very well how a neural network works. And as it was made on Minecraft with redstone I could understand quite easily. 
- After this I jumped to this video : https://www.youtube.com/watch?v=aircAruvnKk&t=303s (the minecraft creator refered to this video for his own creation). 
- After seeing that video I was quite excited about this very new algorithms (to me). 

## First Touch : 
- First of all, I searched for onlines tutorials to create my own neural network. 
- I saw tutorials using Perceptron but this algorithm don't use back propagation, which is one of my favorite "trick" when I saw the video. 
- So I searched again and found a PyTorch tutorial, which I'm going to follow (for now) 
- My knowledge at this point is : 
    - We have inputs. Inputs spreads to the neural network by passing from a layer to an other using maths functions. 
    - At the end of the network they're outputs, which will represent the answer of the neural network (when it'll be trained). 
    - Speeking of training : To train our network we need to use a bank of data which are labelled. We're gonna spread the data through the network, it'll fail, but because data are labelled it'll know what it should have answer, to be more precise, *which* neuronal path is the best to give the correct answer. 
    - It'll adjust the parameters of the maths functions between the layers, depending on a learning rate. I must confess that I don't understand why we can't just put it at max level but I'm sure I'll understand this fast. 

## The Maths Functions : 
- Basically, a neuron is two inputs and an output. 
- Each input has a weight, each neuron has a bias. 
- To determine the output we basically take the input, multiply by his weight, add the two inputs multiplied, and add the bias. 
- We'll also use ReLU function which is very simple. If x is greater than 0 we return x otherwise we return 0. 
- Do this for each neuron of the layer, spread it to the n+1 layer etc. 