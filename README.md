# LeNet5
> "Multilayer Neural Networks trained with the backpropagation algorithm constitute the best example of a successful GradientBased Learning technique."
-----------------
LeNet-5 is considered the first successful practical application of convolutional neural networks. It systematically applied convolutional layers and subsampling (pooling) to achieve outstanding results in handwritten digit recognition. Additionally, it was the first commercial expansion of CNN technology, as it was adopted by banks in the 1990s to automate the reading of handwritten checks using the MNIST dataset as a benchmark.

![image](https://github.com/user-attachments/assets/391759c7-e207-4362-a989-c5551f4aa545)
![image](https://github.com/user-attachments/assets/a8df308b-3268-4867-b61e-1a240efbfe1f)


## Abstract Structure:
![image](https://github.com/user-attachments/assets/ee7dde4d-2ba2-4503-b30a-fba88d41f6bf)

The architecture of the model can be described as having two primary components, each serving a distinct purpose in the overall process of data processing and classification. These components are:

1. **Feature Extraction Module (CNNs)**
2. **Trainable Classifier Module (FCs)**


## Model Structure
![image](https://github.com/user-attachments/assets/d37634b2-9161-4e94-b5e3-c2de7833b39d)

---

### 1. Feature Extraction Module (CNNs)

This module is designed to extract relevant features from raw input data (e.g., images, time-series, or other forms of data) using convolutional neural networks (CNNs). The CNNs are responsible for learning and identifying hierarchical patterns or features at different levels of abstraction. Here’s a breakdown of how the feature extraction module works:

- **Convolutional Layers**:  
  The primary operation in the feature extraction module is the convolution, where filters are applied to the input data to detect low-level features like edges, textures, and patterns. These filters are learned during training, which allows the model to identify features that are most relevant for classification.

- **Activation Functions**:  
  After convolution, an activation function (Sigmoid at this time) is applied to introduce non-linearity.

- **Pooling Layers**:  
  Pooling (typically max pooling) is used to reduce the spatial dimensions of the feature maps, effectively downsampling and reducing the computational complexity while preserving important features.

- **Output**:  
  The output of the CNNs is a set of high-level features that represent the most significant characteristics of the input data. These features are passed to the next stage, the Trainable Classifier Module.

---

### 2. Trainable Classifier Module (FCs)

Once the feature extraction module has processed the input data and extracted the relevant features, the Trainable Classifier Module takes over. This module typically consists of fully connected (FC) layers that perform the actual classification based on the extracted features. Here’s how the classifier module works:

- **Fully Connected Layers**:  
  These layers connect every node in the feature map to every node in the next layer, allowing the network to make non-linear decisions based on the learned features. The fully connected layers combine the high-level features from the CNN module to classify the input data into predefined categories.

- **Output Layer**:  
  The final layer of the FC module is typically a softmax or sigmoid layer, depending on whether the task is multi-class or binary classification. This layer outputs the probability distribution over the possible classes, from which the final prediction is made.

- **Training**:  
  The weights of the FC layers are learned through backpropagation, where the error (difference between predicted output and actual labels) is propagated back to adjust the parameters, improving the model’s ability to classify new data.

### Changes
1. This serves as a proof of concept showing that deep learning can be applied commercially, or real-world applications.
2. Although the model could be built with only FC layers, the paper highlights the importance of using CNNs for feature extraction, enabling better handling of complex data.
3. The general structure, which separates the model into a Feature Extraction Module (CNNs) and a Trainable Classifier Module (FCs), has proven to be effective and has influenced the design of future models.
