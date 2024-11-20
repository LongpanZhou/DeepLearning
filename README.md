# ShuffleNet
> "We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power (e.g., 10-150 MFLOPs)."
-----------------

ShuffleNet achieves a good trade-off between efficiency and accuracy, making it suitable for resource-constrained devices like mobile phones.

Pointwise Group Convolution

![image](https://github.com/user-attachments/assets/8d1fedc0-8306-4dcb-96a8-09274937de8b)

| Depthwise    | Pointwise      |
|--------------|----------------|
| 3x3x3 into 3x1x1. | 3x1 into 1x1.  |

ShuffleNet extensively utilizes pointwise convolution (1x1 convolution) to reduce data size and computational cost, making it well-suited for mobile and resource-constrained devices. This approach efficiently processes feature maps while preserving critical information.

![Channel Shuffle](https://github.com/user-attachments/assets/355da13d-8a49-406a-8ced-a3d29f36e7c5)

This allows it to achieve feature integration comparable to DenseNet but with significantly faster processing and lower computational requirements. It’s a brilliant method for optimizing performance while maintaining efficiency.

![image](https://github.com/user-attachments/assets/188adfd8-f0d8-4420-bfcc-163ba46bbaf1)

When data is split into groups and processed independently, the lack of connection between groups can result in feature loss. In traditional convolution, each input neuron is fully connected to all output neurons, ensuring comprehensive feature integration. To address this, ShuffleNet uses channel shuffle, which reassigns and mixes channels across groups, ensuring connections between input and output nodes are maintained, thereby preserving the integrity of feature representation.

![ShuffleNet_v1](https://github.com/user-attachments/assets/65efd569-86b1-4e7d-b441-9882b179c27c)

Main: Pointwise -> shuffle -> Depthwise -> Pointwise

Side: Depthwise

![ShuffleNet_v2](https://github.com/user-attachments/assets/90b68fd9-d5f8-4dce-b17d-ade87c7fbc4f)

Before branching, the data is split into two streams.

Main: Pointwise -> Depthwise -> Pointwise

Side: Depthwise -> Pointwise

Now it concatenates and shuffles before the activation function.

![Performance](https://github.com/user-attachments/assets/38417b73-32dc-4dbc-89b0-5307f38ebb78)

This graph is for refrence in terms of FLOPs.

![Table](https://github.com/user-attachments/assets/17e2b550-f5fc-4606-a802-6edd38b346e3)

Different configurations of the model are shown here.
