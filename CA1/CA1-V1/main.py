#generated with help of autopilot
from data_processing import load_cifar_datasets, filter_and_combine_datasets
from image_preprocessing import preprocess_images
from model_training import create_and_train_model
from display_histogram import display_histogram
import os


if __name__ == "__main__":
    # Load CIFAR datasets (training and testing data for CIFAR-10 and CIFAR-100)
    cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels, cifar10_test_images, cifar10_test_labels, cifar100_test_images, cifar100_test_labels = load_cifar_datasets()

    # Filter and combine training and testing data from CIFAR-10 and CIFAR-100
    combined_train_images, combined_train_labels = filter_and_combine_datasets(cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels)
    combined_test_images, combined_test_labels = filter_and_combine_datasets(cifar10_test_images, cifar10_test_labels, cifar100_test_images, cifar100_test_labels)

    # Display histograms of label distributions for the combined datasets
    display_histogram(combined_train_labels, combined_test_labels)

    # Preprocess the combined training images (e.g., grayscale, equalization, normalization, and Gaussian blur)
    preprocessed_images = preprocess_images(combined_train_images)

# Check if the model file already exists
if not os.path.exists('model.h5'):
    # Create and train a deep learning model using the preprocessed training data
    model = create_and_train_model(preprocessed_images, combined_train_labels)

    # Save the trained model to a file
    model.save('model.h5')
else:
    print("Model already exists, trying to load and test it!")
