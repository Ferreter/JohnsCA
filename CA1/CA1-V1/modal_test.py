#generated with help of autopilot
import tensorflow as tf
import numpy as np
import random
from data_processing import filter_and_combine_datasets
from image_preprocessing import preprocess_images
from tensorflow.keras.datasets import cifar10, cifar100

def test_model_accuracy(model_path, num_samples=100):
    # Load test datasets
    (_, _), (cifar10_test_images, cifar10_test_labels) = cifar10.load_data()
    (_, _), (cifar100_test_images, cifar100_test_labels) = cifar100.load_data()

    # Combine and preprocess test datasets
    combined_test_images, combined_test_labels = filter_and_combine_datasets(cifar10_test_images, cifar10_test_labels, cifar100_test_images, cifar100_test_labels)
    preprocessed_test_images = preprocess_images(combined_test_images)

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Randomly select images for testing
    test_indices = random.sample(range(len(preprocessed_test_images)), num_samples)
    test_images_sample = np.array([preprocessed_test_images[i] for i in test_indices])
    test_labels_sample = np.array([combined_test_labels[i] for i in test_indices])

    # Evaluate the model
    loss, accuracy = model.evaluate(test_images_sample, test_labels_sample, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_model_accuracy('model.h5')
