# Deep Autoencoder for Fashion MNIST Image Reconstruction

## Overview

This project implements a **Deep Autoencoder** using TensorFlow and Keras to learn compressed representations of Fashion MNIST images and reconstruct them with minimal information loss.

The model is trained in an unsupervised manner where the input images themselves serve as the target outputs. The encoder compresses the images into a low-dimensional latent space, while the decoder reconstructs the original images from these compressed representations.

---

## Dataset

**Fashion MNIST**

Fashion MNIST is a dataset of Zalando's article images consisting of:

* 60,000 training images
* 10,000 testing images
* 10 clothing categories
* Grayscale images of size 28×28 pixels

Each image is flattened into a vector of 784 features before being fed into the autoencoder.

---

## Model Architecture

### Encoder

Input Layer (784)

→ Dense (512, ReLU)

→ Dense (256, ReLU)

→ Dense (128, ReLU)

→ Dense (64, ReLU) *(Latent Representation)*

### Decoder

Dense (128, ReLU)

→ Dense (256, ReLU)

→ Dense (512, ReLU)

→ Dense (784, Sigmoid)

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Kaggle Notebook Environment

---

## Data Preprocessing

1. Load Fashion MNIST dataset.
2. Normalize pixel values to range [0,1].
3. Flatten 28×28 images into 784-dimensional vectors.
4. Train the autoencoder using input images as both features and labels.

```python
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train = X_train.reshape((len(X_train), 784))
X_test = X_test.reshape((len(X_test), 784))
```

---

## Training Configuration

| Parameter          | Value               |
| ------------------ | ------------------- |
| Optimizer          | Adam                |
| Loss Function      | Binary Crossentropy |
| Batch Size         | 256                 |
| Epochs             | 10                  |
| Encoding Dimension | 64                  |

---

## Results

After training:

* The autoencoder successfully learns compressed image representations.
* Reconstructed images closely resemble the original Fashion MNIST images.
* Training and validation loss consistently decrease over epochs.
* The latent space captures essential visual information while reducing dimensionality from 784 features to 64.

---

## Visualization

### Original vs Reconstructed Images

The project visualizes:

* Original Fashion MNIST images
* Autoencoder reconstructed images

allowing qualitative evaluation of reconstruction performance.

### Learning Curves

The following metrics are plotted:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

to monitor model convergence.

---

## Author

**Gunjan Soni**

Computer Science Student | AI/ML Enthusiast | Deep Learning Explorer

---
