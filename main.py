import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()

training_images, testing_images = training_images / 255.0, testing_images / 255.0 # divide images by 255.0 to scale pixel values between 0 and 1

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()
