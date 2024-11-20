# DenseNet
> "In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion."
-----------------
**What if we copy resnet's homework?**

DenseNet introduces shortcuts to each layer's activation function, enabling the neural network to achieve massive depth (with 121 layers as the minimum). While this significantly increases computational cost, but with great power comes great responsiblity. 😈

## Key Points

Densely Connected Layers

![image](https://github.com/user-attachments/assets/0749cb42-a5f1-44cd-a599-11ae392154cd)

As shown in the image, in DenseNet, each layer is short-circuited and connected to every other layer. This dense connectivity allows each layer to receive input from all previous layers, improving feature reuse and enhancing the network's ability to learn complex representations.

## Model Structure
![image](https://github.com/user-attachments/assets/79c63e3b-2c21-43c1-82c2-0552464adb31)

The model's structure consists of multiple densely connected blocks, with each block separated by convolutional and pooling layers. This design allows for efficient feature reuse and deep learning while maintaining manageable computational complexity through strategic pooling and convolution operations between the blocks.

![image](https://github.com/user-attachments/assets/5c3729cd-074e-4f9b-8b37-f8aee866af78)

Like ResNet, it also offers different configurations of the network.

![image](https://github.com/user-attachments/assets/f696ed72-08ae-461c-9975-025afc11628f)

Results shows a way better results for densenet compare to resnet.

## Changes
1. Very deep layers — DenseNet can have extremely deep networks, with the smallest configuration having 121 layers, enabling the model to capture more complex features.
