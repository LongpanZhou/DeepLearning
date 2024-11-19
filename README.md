# GoogLeNet
> "The main hallmark of this architecture is the improved utilization
of the computing resources inside the network. This was achieved by a carefully
crafted design that allows for increasing the depth and width of the network while
keeping the computational budget constant."
-----------------

GoogLeNet's standout feature is its efficient computation, achieved by using Inception Blocks to process features at multiple scales and by breaking down larger convolutions into smaller ones. Another key innovation is the bottleneck layer, which reduces dimensionality and computation, a technique still widely adopted in neural networks for mobile and lightweight applications.

## Key Points
**Pictures from online resources**

Inception Block

![image](https://github.com/user-attachments/assets/b50ba4a7-d9d5-4b39-abde-17a396ee4268)

The Inception Block reduces computational cost by dividing the input data into multiple branches, each performing different convolutions or pooling operations in parallel. These outputs are then combined, enabling efficient feature extraction across multiple scales while keeping the model computationally efficient.

Decomposition

![Screen-Shot-2018-04-17-at-5 32 45-PM](https://github.com/user-attachments/assets/7146647d-1cd8-4a8e-80fc-c322661e147e)

As shown in the image, a 5x5 convolution can be decomposed into two consecutive 3x3 convolutions. This approach significantly reduces computational cost by lowering the number of parameters and operations required, while still maintaining the same receptive field size.  By calculation two 3x3 convolutions are approximately 39% faster than a single 5x5 convolution in terms of operations.

BottleNeck Layer

![image](https://github.com/user-attachments/assets/37e29f8b-83d5-438c-8991-ddbaa414e54b)

The BottleNeck layer is designed to reduce computational cost in the hidden layers of a network. As the name suggests, the hidden layer has a smaller number of channels compared to the input and output channels, often reduced by a factor of 4. This reduction minimizes the number of computations while preserving the model's ability to learn effective representations. This technique remains widely used in modern neural networks, especially in lightweight architectures for mobile and embedded systems.

Multiple Outputs

From pytorch
```
def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        #codes...
        return x, aux2, aux1
```

The reason why it has multiple outputs is because altough it reduced the computational cost, but the problem of dimishing graident has still not been solved yet. By introducing mutiple outputs/aux, it can use the aux as a starting point for back praprogation.

By introducing auxiliary classifiers (e.g., aux1, aux2) as intermediate outputs, the network can compute additional gradients from these auxiliary branches. These gradients act as a "boost" to the earlier layers, effectively serving as secondary starting points for backpropagation. This technique stabilizes training and improves gradient flow in very deep architectures.

## Model Structure
![image](https://github.com/user-attachments/assets/6c3ccaf1-f284-4ec5-82cb-6f998bde4e55)
![image](https://github.com/user-attachments/assets/627de117-638e-4dc8-b83f-c024420b9255)

![image](https://github.com/user-attachments/assets/e455ea85-839c-4f29-96bb-f7679f9e739e)

Yeah, I know. Let's look at the one below!

![inception-full-90](https://github.com/user-attachments/assets/75fc937f-0c34-416e-9175-f4a524332c61)

The implementation of GoogLeNet is pretty straight forward, the question comes from why did they make the decisions of [2,5,3] inception blocks in between. (I do not know.)

## Changes
1. Introduced branching within blocks and concatenation of outputs, allowing the network to learn multi-scale features efficiently.
2. Reduced computational cost by varius of different convolutions techiniques, allowing CNN networks on mobile devices possible.
