#generated with help of autopilot
import cv2
import numpy as np

def preprocess_images(images):
    #Convert images to grayscale
    gray_images  = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    #Equalize the histogram of grayscale images for better contrast
    equalized_images = [cv2.equalizeHist(img) for img in gray_images]
    #Reshape the images to the desired shape (e.g., (32, 32, 1))
    images_array = np.array(equalized_images).reshape((-1, 32, 32, 1))
    #Normalize pixel values to the range [0, 1]
    images_normalized = images_array.astype('float32') / 255.0
    #Apply Gaussian blur to the normalized images
    blurred_images = np.array([cv2.GaussianBlur(img, (5, 5), 0) for img in images_normalized])

    return blurred_images