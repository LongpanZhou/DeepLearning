# MobileNet
> "MobileNets are based on a streamlined architecture that uses depthwise separable convolutions to build light weight deep neural networks. We introduce two simple global hyperparameters that efficiently trade off between latency and accuracy. "
-----------------

MobileNets is one of the earliest neural network architectures specifically designed for lightweight models, ideal for mobile and embedded devices where computational resources are limited. It consists of three versions: V1, V2, and V3, each version of MobileNet builds upon its predecessors, offering significant improvements in speed and accuracy, making them suitable for real-time applications on devices with limited processing power.

Since the resources of MobileNet are too extensive to cover in one section, they are divided as follows:

| MobileNet Version                     | Description                                                                                               |
|---------------------------------------|-----------------------------------------------------------------------------------------------------------|
| [MobileNet v1.md](./MobileNet/MobileNet_v1.md) | Introduces depthwise separable convolutions to reduce model size and computational cost.                 |
| [MobileNet v2.md](./MobileNet/MobileNet_v2.md) | Features inverted residual blocks with linear bottlenecks for improved efficiency and performance.         |
| [MobileNet v3.md](./MobileNet/MobileNet_v3.md) | Optimized with H-Swish activation and improved architecture via Neural Architecture Search (NAS) for optimal balance between speed and accuracy. |


It explains each version of the MobileNets in a more detailed description of what each version includes and how it improves upon the previous one.
