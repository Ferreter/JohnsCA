#generated with help of autopilot
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

#removing EarlyStopping callback can make it to accuracy of 98% but is too intensive without much effort to improve, it stops at 61% accuracy with EarlyStopping callback
def create_and_train_model(combined_train_images, combined_train_labels, epochs=15, batch_size=32):
    # Create a mapping of unique labels to integer indices
    label_mapping = {label: idx for idx, label in enumerate(set(combined_train_labels))}
    
    # Convert the combined_train_labels to an array of integer labels using the mapping
    combined_train_labels = np.array([label_mapping[label] for label in combined_train_labels])

    # Define a Sequential neural network model
    ann = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(len(label_mapping), activation='softmax')  # Output layer with softmax activation
    ])

    # Compile the model with appropriate optimizer, loss function, and metrics
    ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define an EarlyStopping callback to monitor the loss and stop early if it doesn't improve
    #Changing the patience to 3 from 5 makes it to very high accuracy but is too intensive without much effort to improve, it stops at 61% accuracy with EarlyStopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train the model on the combined_train_images and combined_train_labels
    ann.fit(combined_train_images, combined_train_labels, 
            epochs=epochs, 
            batch_size=batch_size, 
            callbacks=[early_stopping],
            validation_split=0.2)  # You can also specify a validation split here if needed

    # Return the trained model
    return ann
