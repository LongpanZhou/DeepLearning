# Deep Learning
> [!IMPORTANT]  
> This repo includes all the milestone models in deep learning, featuring the implementation of each model along with a summary that highlights what makes each model unique and how it advances the field compared to previous architectures. If you wish to follow a path and build your own models from the most basic until most modern, this will hopefully be a good resources for you.

> [!NOTE]  
> As the models become more complex, training them on a personal computer may become challenging due to hardware limitations. In such cases, I strongly recommend using a computing cluster or cloud services to facilitate training.
---
## DataSet
This project uses [ImageNet100](https://www.kaggle.com/datasets/ambityga/imagenet100) (16.4GB) - a subset of ImageNet with 100 classes, for training and benchmarking models.

Download Link:
```
https://storage.googleapis.com/kaggle-data-sets/1500837/2491748/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241119%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241119T032223Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9627a949319acd5429ce0dd2a00abd57f0633dc700701011ec65651842cf4553080e29e5d30b8bb76520de6c29213f7344a278786161ba61c41e79c21a3fd3c96bdc5ae072e39fe386a3f52c290ca7d3423abfead746cf8299d82cae6f7f6b9d0212e40d6fa1a0bd643062680866677feebbc833b790a7ab2b068aa7a41800c4aaeab168953e92152ee5e22b539f5b59700a1726f9e8be5202669ac720a8a390dac6180c05fc1c5985d31e897664e2b19294f070a0395fa24c041d86ed6e91b1bfa64c38ee1e4381e3ad8ed949ade3fd4e82992bac4a20aa381706bfec1f4fd53884cf0e2f30ea70c0926157a794877b029a4e250829bd6afa8709c102fc24ee
```

![image](https://github.com/user-attachments/assets/4181f327-bafe-4d48-930d-6fe33559a392)


## Models

All the models are under different branches - current: main

### 1. [LeNet](https://github.com/LongpanZhou/DeepLearning/tree/LeNet)
- **Paper**: [GradientBased Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- **DigitalOcean**: [https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py](https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python)

### 2. [AlexNet](https://github.com/LongpanZhou/DeepLearning/tree/AlexNet)
- **Paper**: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- **PyTorch**: https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py

### 3. [VGG](https://github.com/LongpanZhou/DeepLearning/tree/VGG)
- **Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)
- **PyTorch**: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

### 4. [GoogLeNet](https://github.com/LongpanZhou/DeepLearning/tree/GoogLeNet)
- **Paper**: [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842)
- **PyTorch**: https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py

### 5. [ResNet](https://github.com/LongpanZhou/DeepLearning/tree/ResNet)
- **Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)
- **PyTorch**: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

### 6. [DenseNet](https://github.com/LongpanZhou/DeepLearning/tree/DenseNet)
- **Paper**: [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993)
- **PyTorch**: https://github.com/liuzhuang13/DenseNet

### 7. [MobileNet](https://github.com/LongpanZhou/DeepLearning/tree/MobileNet)
- **Paper**: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861)
- **Paper**: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381)
- **Paper**: [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244)

### 8. [ShuffleNet](https://github.com/LongpanZhou/DeepLearning/tree/MobileNet)
- **Paper**: [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083)
- **Paper**: [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/pdf/1807.11164)

**UNDER DEVELOPMENT**

EffeientNet, ViT, DeiT, Swin, ConvNet, InceptionNeXt, Mamba (Not sure if I will implement all)

---

## Innovations
- [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
- [Deep Learning using Rectified Linear Units (ReLU)](https://arxiv.org/pdf/1803.08375)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)

![Meme](https://github.com/user-attachments/assets/24dd26c3-2b1c-486c-9cb4-93fdc19c5b46)
