# VGG
> "Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with
very small (3 × 3) convolution filters, which shows that a significant improvement
on the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers."
-----------------
**VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION** - as the title suggest this is a model which present a deep convolutional network architecture, which serves as a successor to AlexNet. The primary focus of the model is exploring the impact of increasing the depth of convolutional neural networks (CNNs) on large-scale image recognition tasks.

*Brute force leads to miracles* - MY RTX3080 SPEND 30MINS! ON 10 EPOCHS!

## Model Structure
![vgg](https://github.com/user-attachments/assets/e96c3692-7a2f-463b-9b20-0ecf9672fd61)

The VGG network uses VGG blocks to construct its architecture. Compared to AlexNet, there is no significant increase in the number of parameters, but the model's depth is greater. Increasing the depth of the network improves accuracy by enabling it to learn more complex features.

```
def VGG_block(self, in_channels, cfg, batch_norm=False):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
```

Each VGG block consists of a series of 2D convolutions, followed by ReLU activation functions, and ends with a MaxPool layer to reduce spatial dimensions and enhance feature extraction. Compared to AlexNet, VGG uses 3x3 convolutions or 1x1 convolutions for feature extraction, a design choice that became popular and contributed to improved accuracy. Additionally, the introduction of Batch Normalization (not used in the original paper) can significantly speed up convergence and mitigate issues like vanishing or exploding gradients.

![image](https://github.com/user-attachments/assets/c011a956-2fe2-4373-94da-9dcad4b23159)

VGG was also the first network to introduce different configurations of the network, allowing flexibility in architecture.
```
self.block_config  = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
```

## Changes
1. Smaller convolution filters (3x3) improve performance by capturing finer details while reducing computational cost.
2. Deeper networks capture more abstract features, leading to better performance on tasks like image recognition.
3. CNN blocks, consisting of stacked convolutions, activations, and pooling, help the model efficiently extract hierarchical features.
4. Different model configurations.

## Problems
1. The diminishing backpropagation problem - As networks deepen, gradients shrink during backpropagation, making training harder for earlier layers.
2. The curse of computation - Deeper networks with more parameters demand significant computational resources and longer training times.
