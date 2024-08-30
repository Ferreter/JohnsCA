#generated with help of autopilot
import matplotlib.pyplot as plt

def display_histogram(train_labels, test_labels, title="Dataset Label Distribution"):
    # Create a figure with two subplots (side by side).
    plt.figure(figsize=(10, 5))

    # Plotting histogram for train labels in the first subplot.
    plt.subplot(1, 2, 1)
    plt.hist(train_labels, bins=len(set(train_labels)), color='blue', alpha=0.7)
    plt.title('Train Labels')   # Set the title for the train labels histogram.
    plt.xlabel('Label')         # Set the x-axis label.
    plt.ylabel('Frequency')     # Set the y-axis label.

    # Plotting histogram for test labels in the second subplot.
    plt.subplot(1, 2, 2)
    plt.hist(test_labels, bins=len(set(test_labels)), color='green', alpha=0.7)
    plt.title('Test Labels')    # Set the title for the test labels histogram.
    plt.xlabel('Label')         # Set the x-axis label.
    plt.ylabel('Frequency')     # Set the y-axis label.

    # Set the main title for the entire figure.
    plt.suptitle(title)

    # Adjust the layout to prevent subplot overlap and display the figure.
    plt.tight_layout()
    plt.show()
