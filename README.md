# SENet
> "In this work, we focus instead on the channel relationship and propose a novel architectural unit, which we term the “Squeeze-and-Excitation” (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels."
-----------------
Squeeze-and-Excitation works by first applying a global average pooling operation to squeeze spatial information and then using a fully connected layer to excite the features across channels. This recalibration improves the network’s ability to capture important channel-wise dependencies, leading to better performance in tasks such as image classification.
## Key Points
RGB Decomposition

![image](https://github.com/user-attachments/assets/81f171b4-47eb-4e5f-8eb2-270d7acf2d4c)

Images are composed of three color channels: Red, Green, and Blue (RGB). Each channel represents the intensity of that color in the image, and by decomposing the image into its individual RGB components, we can analyze and manipulate the color information separately. This is fundamental for many computer vision tasks, allowing the network to learn and process the color-based features more effectively.

An shark example picture from ImageNet 100

![n01494475_3562](https://github.com/user-attachments/assets/2c0b076f-a522-4a0e-a3d7-5886e44c9823)

When classifying sharks, the blue color in the image will often be emphasized in the model's feature extraction process. The model may use this information, along with other features, to distinguish sharks from other objects in the image, leveraging the interdependencies between color channels and spatial features to make an accurate classification.

## Block Structure
SE-Block

![image](https://github.com/user-attachments/assets/a6a164ae-acc1-4276-aac1-a09020cc91ef)

The SE_Block adjusts the contribution of each channel in the feature map dynamically, allowing the network to focus more on relevant channels and suppress irrelevant ones, thus improving the model's representational power and performance.

SE-Inception Block

![Figure_2](https://github.com/user-attachments/assets/73eab099-536e-4559-bd9a-b8e9af4146df)

SE-ResNet Block

![Figure_3](https://github.com/user-attachments/assets/e9ad77b0-adfa-4053-92e1-a18f2d3e2c3b)
