# Cat vs Dog Classification using CNN with InceptionV3

This project is a binary classification task where the goal is to classify images as either a cat or a dog. We use a Convolutional Neural Network (CNN) with the InceptionV3 architecture, leveraging the power of transfer learning.

## Project Overview

The project uses PyTorch as the main library for implementing the model. The InceptionV3 model, pre-trained on the ImageNet dataset, is used as a base model. The final layers of the model are fine-tuned to classify images as either a cat or a dog.

The model is trained using a custom dataset of cat and dog images. The dataset is divided into a training set and a test set, with the training set used to train the model and the test set used to evaluate its performance.

The model is trained using a batch size of 32 and a learning rate of 0.001 for 1 epoch. The model's weights are updated using the Adam optimizer, and the loss is calculated using Cross-Entropy Loss.


## Results

The model achieves an accuracy of around 95% on both the training and test datasets. The quantized model with 8bit floats, maintains a similar level of accuracy while significantly reducing the model size and computational requirements.

## Future Work

Future work on this project could include experimenting with different model architectures, training for more epochs, or using different techniques for data augmentation to further improve the model's performance.