# cassava_disease_classification
This is a  [kaggle](https://www.kaggle.com/competitions/ammi-2024-computer-vision/overview) competition. The aim of the competition is to help cassava farmers identify healthy and infected cassava by observing the leaves. Cassava is one of the most grown crop by small scale farmers in Africa and by classifying the leaves it would help in improving food security among the small holder farmers. 

The dataset used which can be obtained [here](https://www.kaggle.com/competitions/ammi-2024-computer-vision/data), has leaf images of the cassava plant, with 9,436 annotated images and 12,595 unlabeled images of cassava leaves. In this project we only used the  9,436 annotated images. The goal is to learn a model to classify a given image into these 4 disease categories or a 5th category indicating a healthy leaf, using the images in the training data.

In this project, we divided the training data into two, 80\% for train and 20\% for validation. 

# Models Used
We built two pretrained convolutional neural network (CNN) models, that is, Inception_v3 and Resnet50 to train our dataset. 

## Inception_V3
* It is a deep convolutional neural network (CNN) architecture designed for image classification tasks. It is an improvement over earlier versions of the Inception architecture, specifically InceptionV1 and V2, and was developed by Google. The model is known for its efficiency in using computational resources while maintaining high accuracy. 
* **Architecture: InceptionV3 uses a combination of multiple convolutional layers, including 1x1, 3x3, and 5x5 filters, as well as pooling layers, to capture spatial hierarchies at different scales. It introduces the idea of factorized convolutions, where large convolutions (e.g., 5x5) are split into smaller ones (e.g., two 3x3 convolutions) to reduce computational cost.
*** Applications:** InceptionV3 is widely used in image classification, object detection, and transfer learning tasks due to its balance between accuracy and computational efficiency.

## ResNet5O
* It introduced the concept of skip connections.
* **Architecture:** ResNet50 is composed of 50 layers, including convolutional layers, batch normalization, and ReLU activations. Its key innovation is the use of "skip connections" or "identity shortcuts," which allow the model to bypass one or more layers. This addresses the problem of vanishing gradients, enabling the training of very deep networks.
* **Applications:** ResNet50 is highly effective in image classification tasks and is commonly used for feature extraction, transfer learning, and various computer vision applications due to its depth and accuracy.

# Training
* Preprocessing: Before running the models we performed preprocessing by augmenting the dataset.
* Optimization: We used stochastic gradient descent as our optimizer. The SGD used momentum of 0.99 and learning rate of 0.0001 for the inception_v3 model. As for the Resnet50 we used a momentum value of 0.9 and a learning rate of 0.001.



