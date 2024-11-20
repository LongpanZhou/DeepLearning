# MobileNet v2

MobileNetV2 introduces the concept of inverted residual blocks, which improves computational efficiency and model performance by utilizing lightweight depthwise separable convolutions and applying linear bottleneck layers.

![Relu Loss](https://github.com/user-attachments/assets/da489f33-1da5-43e0-b507-7d04d033fd16)

The image illustrates how the ReLU function can lead to information loss, particularly when the dimensionality is low. 

ReLU (Rectified Linear Unit) can cause information loss because it outputs zero for all negative values, which means any negative input to the activation function is completely discarded. When the network encounters low-dimensional data or features, the number of negative values may increase, leading to a significant loss of information. This issue is more prominent when the network is shallow or when there's insufficient data to activate all neurons effectively.
Also known as the "Dead Neuron Problem".

Using linear convolutions (or a linear activation function applied after convolutions) can be better than using ReLU in certain situations, particularly in lower-dimensional spaces or when dealing with specific tasks that require more subtle transformations. Here's why:

![Linear Bottleneck](https://github.com/user-attachments/assets/9ff280f4-0212-423e-aafc-abd0c2384d40)

- Regular: The standard convolution operation, where each filter processes all input channels, offering general feature extraction.
- Separable: Depth-wise and point-wise convolutions, improving computational efficiency by reducing the number of parameters.
- Separable with Linear Bottleneck: Utilizes linear activation in the bottleneck layer, reducing information loss while maintaining efficiency.
- Bottleneck with Expansion Layer: Expands the number of channels before applying a bottleneck, allowing the model to learn richer features while maintaining computational efficiency.

![Inverted Residuals](https://github.com/user-attachments/assets/15bb3893-27ee-410b-a047-a0cddddecfca)

| **Block Type**           | **1x1 Conv (Initial)** | **3x3 Conv (Middle)**      | **1x1 Conv (Final)**        | **Dimension Flow**          |
|--------------------------|------------------------|----------------------------|-----------------------------|-----------------------------|
| **Residual Block**        | Reduces                | Remains                    | Increases                   | Reduce → Remain → Increase  |
| **Inverted Residual Block** | Increases (with expansion ratio) | Remains                    | Reduces                    | Increase → Remain → Reduce  |


![Conv Block](https://github.com/user-attachments/assets/b70b5bed-4ade-4f07-9860-770f1318c620)

The convolution block uses shortcuts with stride 1 to improve gradient flow and accuracy, while skipping them with stride 2 to avoid redundancy and focus on downsampling (Some what like ResNet) balancing computational cost and accuracy. 

![Bottleneck Conv](https://github.com/user-attachments/assets/6661d9e0-6580-4504-8176-37a8bc0bbcda)

Notice the last layer is being replaced by linear 1x1 conv instead of ReLU function as dicussed before.

![Structure](https://github.com/user-attachments/assets/6ddd968d-84b1-4dbe-902c-b81197d339a2)
