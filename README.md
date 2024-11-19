# AlexNet
> "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes."
-----------------
In the 2012 ImageNet competition, AlexNet achieved a groundbreaking result. Named after its creator, Alex Krizhevsky, along with Ilya Sutskever (OpenAI CTO) and Geoffrey E. Hinton (Turing Award 2019, Physics Nobel Prize 2024), it marks a continuation of the success that CNNs had already begun to achieve.

## Abstract Structure:
![image](https://github.com/user-attachments/assets/d9cd225e-fce9-4043-bcfe-90fa035992ac)

The architecture of AlexNet is based on LeNet's design, where convolutional layers (CNNs) are followed by fully connected layers (FCs). A key distinction in AlexNet's structure is the division of the model into two parts, as shown in the graph. This split occurs because the neural network is being trained across two GPUs, which enables more efficient parallel processing and handling of large-scale computations.

## Model Structure
![alexnet](https://github.com/user-attachments/assets/43197104-60ff-49b2-9718-e2de114034e5)

LeNet (Left) To AlexNet(Right)

The fundamental difference between LeNet and AlexNet is that AlexNet has significantly more parameters in both the CNN and FC layers compared to LeNet. This increase in parameters is a result of advancements in computing power and the availability of larger datasets, which made it possible to train much deeper networks. These developments allowed AlexNet to capture more complex patterns and achieve superior performance on tasks like image classification.

## Changes
1. Deep learning has evolved into a "black box" technology, where the inner workings of models are often too complex to fully interpret or understand.
2. The deeper the CNN, the better its performance tends to be, as adding more layers allows the model to capture increasingly abstract features from the data.
3. The use of GPUs for training neural networks has become essential, as they significantly speed up the process by handling parallel computations more efficiently than CPUs.
4. ReLU activation functions are widely used in deep learning models because they help mitigate the vanishing gradient problem, enabling faster convergence and better performance, especially in deeper networks.
