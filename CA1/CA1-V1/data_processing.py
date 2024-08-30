#generated with help of autopilot
import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100

def load_cifar_datasets():
    (cifar10_train_images, cifar10_train_labels), (cifar10_test_images, cifar10_test_labels) = cifar10.load_data()
    (cifar100_train_images, cifar100_train_labels), (cifar100_test_images, cifar100_test_labels) = cifar100.load_data()
    return cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels, cifar10_test_images, cifar10_test_labels, cifar100_test_images, cifar100_test_labels

def filter_and_combine_datasets(cifar10_train_images, cifar10_train_labels, cifar100_train_images, cifar100_train_labels):
    cifar10_classes = {'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'horse': 7, 'truck': 9}
    cifar100_classes = {'cattle': 11, 'fox': 34, 'baby': 2, 'boy': 11, 'girl': 35, 'man': 44, 'woman': 98,
                        'rabbit': 65, 'squirrel': 78, 'trees': 84, 'bicycle': 8, 'bus': 13,
                        'motorcycle': 48, 'pickup truck': 58, 'train': 95, 'lawn-mower': 48, 'tractor': 86}

    def filter_dataset(images, labels, class_mapping):
        filtered_images = []
        filtered_labels = []
        for img, lbl in zip(images, labels):
            class_id = lbl[0]
            if class_id in class_mapping.values():
                class_name = list(class_mapping.keys())[list(class_mapping.values()).index(class_id)]
                filtered_images.append(img)
                filtered_labels.append(class_name)
        return filtered_images, filtered_labels

    cifar10_filtered_images, cifar10_filtered_labels = filter_dataset(cifar10_train_images, cifar10_train_labels, cifar10_classes)
    cifar100_filtered_images, cifar100_filtered_labels = filter_dataset(cifar100_train_images, cifar100_train_labels, cifar100_classes)

    combined_images = cifar10_filtered_images + cifar100_filtered_images
    combined_labels = cifar10_filtered_labels + cifar100_filtered_labels

    return combined_images, combined_labels
