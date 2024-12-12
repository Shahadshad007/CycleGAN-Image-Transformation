# Horse2Zebra Dataset for CycleGAN

This repository contains code for training a **CycleGAN** model to perform image translation between **horses** and **zebras** using the **Horse2Zebra** dataset. CycleGAN is a type of Generative Adversarial Network (GAN) that learns to map images from one domain to another without requiring paired training data.

## Dataset Overview

The **Horse2Zebra** dataset is part of the collection of datasets for CycleGAN and consists of images of horses and zebras. The dataset is unpaired, meaning there are no corresponding images between the two classes. The goal of the model is to learn a transformation from one domain (horse images) to another (zebra images), and vice versa.

- **Dataset A (Horse)**: Images of horses.
- **Dataset B (Zebra)**: Images of zebras.

### Dataset Location

You can download the Horse2Zebra dataset from [this link](datasets). Extract the dataset into the `horse2zebraA` and `horse2zebraB` folders in the directory.

## Model Overview

### CycleGAN

CycleGAN is a framework that learns to translate images from one domain to another without paired examples. It consists of two generators and two discriminators:

- **Generator AB**: Transforms images from the horse domain to the zebra domain.
- **Generator BA**: Transforms images from the zebra domain to the horse domain.
- **Discriminator A**: Evaluates images from the horse domain.
- **Discriminator B**: Evaluates images from the zebra domain.

Cycle consistency loss is used to ensure that the generated images can be converted back to the original domain.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL

To install the necessary dependencies, you can create a virtual environment and install the requirements using:

```bash
pip install -r requirements.txt
