# ResNet
> "We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions."
-----------------
ResNet is groundbreaking research with its core idea of shortcut connections that linking the input of a layer directly to its activiation function. In the paper, each pair of consecutive layers is connected through these shortcuts, which effectively halves the depth of the model in terms of gradient flow. This allows for more effective backpropagation, addressing the vanishing gradient problem and enabling the training of much deeper networks.

## Key Points

Residual Block

![residual-block](https://github.com/user-attachments/assets/82361558-f491-46fe-a22b-e25c624d4aa7)

The structure of the residual block is simple yet effective: it links the input of a layer directly to the output after the layer's processing, bypassing the activation function. During backpropagation, the gradients are also propagated directly through this shortcut connection.

(56-Layers Plain Blocks Vs 20-Layers Residual Blocks)

![image](https://github.com/user-attachments/assets/6eb565d7-1183-4b69-a996-2af38bb4bd67)

This leads to faster convergence, more accurate training results, and better overall performance with lower computational costs, allowing the training of deeper networks without sacrificing efficiency.

![resnet-block](https://github.com/user-attachments/assets/4d12e6dd-0702-41e0-8c64-426007bab637)

The model uses a mix between these two above blocks.

## Model Structure

![resnet18-90](https://github.com/user-attachments/assets/9e92e684-6455-41f1-ba69-8e3c5a5c648e)

Unlike GoogLeNet, where blocks are branched and execute different convolutions in parallel, ResNet uses a continuous block structure where each block is connected in sequence. And there is no maxpool in between, each block effectivly feeds its outputs to the next.

![image](https://github.com/user-attachments/assets/a00800e3-8f8c-41fb-b081-9c43bda22806)

Like VGG, ResNet also provides different configurations of the network to fit for different use cases. This can be done easily when constructing the model by looping through its configuration.

![image](https://github.com/user-attachments/assets/59666f60-924c-4f3f-b1ea-c071f86fa71a)

ResNet "Smokes" other models as shown in the table.

## Changes
1. **Short-Cuts** enabled the training of deeper neural networks by addressing the vanishing gradient problem. *-Major Breakthrough*
