# MobileNet v3

MobileNetV3 optimized the architecture even more by incorporating automated neural architecture search (NAS) to find the most efficient architecture, achieving better speed and accuracy than its predecessors, making it ideal for mobile and edge devices.

![bneck](https://github.com/user-attachments/assets/b42dd38d-a0a1-49bf-9641-cae844a38547)

There is not major change in bneck block, except for adding Squeeze-and-Excitation block. They have also reduced the kernal size from 32 to 16, for the first convolution layers compare to last.

![Last Stage](https://github.com/user-attachments/assets/7dfdc77d-76dd-451f-89a9-1f7ac16bd1aa)

The original last stage was determined using NAS(some fancy brute force), but later testing found that the Efficient Last Stage structure could reduce some redundant layers without sacrificing accuracy. And the model uses H-Swish activiation function instead of ReLU.

![MobileNetv2_large](https://github.com/user-attachments/assets/b92006a2-8737-4a03-b37b-36dde3d9452f)

![MobileNetv2_small](https://github.com/user-attachments/assets/0d43b93b-4633-46d8-b2b8-8ec107c2be47)

The configuration of the model's implementation can be seen above.

![Comparision](https://github.com/user-attachments/assets/3a5c1de9-57a9-4b53-b534-4ff733dded68)
