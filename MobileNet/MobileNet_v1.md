# MobileNet v1

MobileNetV1 introduced depthwise separable convolutions to reduce the computational complexity compared to standard convolutions, which helped make it more efficient while maintaining good performance for image classification tasks.

![Conv](https://github.com/user-attachments/assets/33c2a4b8-944d-4766-b13a-747d28359c7e)

MobileNets uses depthwise separable convolutions to reduce computation. First, depthwise convolution processes each channel separately, and then a 1x1 pointwise convolution combines the outputs. This technique reduces computation by 8-9 times with minimal loss in accuracy, making it ideal for mobile and edge devices.

![MboileNets_v1_block](https://github.com/user-attachments/assets/1165a5bb-e1b1-450f-8de1-b1bbffdfddbf)

A standard MobileNet block, as seen above, consists of the following steps:
- **Depthwise Convolution**: Applies a separate filter for each input channel, reducing computational cost.
- **Pointwise Convolution** (1x1 Convolution): Combines the outputs of the depthwise convolution, capturing cross-channel dependencies.

![Computation breakdown](https://github.com/user-attachments/assets/cbacc05c-d6bc-4222-9abf-6ee3bd4e062f)

The computation brokendown of MobileNet.

![Conv Vs Depth Conv](https://github.com/user-attachments/assets/47de7779-2be3-4693-8e52-627dec4ba9f7)

The amount of computation between standard convolution and depthwise separable convolution is dramatically reduced.

![Conv_MobileNet Vs MobileNet](https://github.com/user-attachments/assets/ff43d7a1-b8b6-4fb4-8088-525928028d59)

As shown in the image, the computational cost is significantly reduced with only a 1% decrease in accuracy.

![MobileNetv1](https://github.com/user-attachments/assets/a53a5ba4-3351-48fc-abc6-c3313dd29697)

The configuration of the MobileNet v1.
