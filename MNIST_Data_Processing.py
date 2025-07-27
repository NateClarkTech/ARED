"""# Create Skewed MNIST"""

import numpy as np
import pickle
import os
from sklearn.datasets import fetch_openml

# Create skewed subset
def create_skewed_mnist(X, y, sparsity_levels, n_events):
    np.random.seed(42)
    digit_order = np.random.permutation(10)
    indices = []
    for digit, count in zip(digit_order, sparsity_levels):
        digit_indices = np.where(y == str(digit))[0]
        if len(digit_indices) >= count:
            selected = np.random.choice(digit_indices, count, replace=False)
            indices.extend(selected)
        else:
            print(f"Warning: Not enough samples for digit {digit}, using all {len(digit_indices)}")
            indices.extend(digit_indices)
    indices = np.array(indices)
    np.random.shuffle(indices)
    if len(indices) > n_events:
        indices = indices[:n_events]
    X_skewed = X[indices]
    y_skewed = y[indices]
    return X_skewed, y_skewed

def load_and_skew_mnist(sparsity_levels, n_events, save_path="mnist_full.pkl"):
    """
    Loads MNIST, creates a skewed subset using create_skewed_mnist.
    Also saves the full MNIST (X, y) to a pickle file.

    Args:
        sparsity_levels: list of 10 integers, number of samples per digit.
        n_events: total number of samples to include in final skewed dataset.
        save_path: path to store full MNIST data as pickle.

    Returns:
        X_skewed, y_skewed: filtered and shuffled MNIST subset.
        X, y: full MNIST dataset.
    """
    if os.path.exists(save_path):
      # Load the data from the .pkl file
      with open(save_path, "rb") as file:
        X,y = pickle.load(file)
      print("Data loaded successfully:")
    else:
      print("Loading MNIST from OpenML...")
      mnist = fetch_openml("mnist_784", version=1, as_frame=False)
      X, y = mnist.data, mnist.target  # y is a string array of digits
      print(f"Full MNIST loaded: {X.shape[0]} samples")
      # Save full dataset
      with open(save_path, "wb") as f:
          pickle.dump((X, y), f)
      print(f"Full MNIST (X, y) saved to {save_path}")

    # Create skewed subset
    print(f"Creating skewed subset with sparsity {sparsity_levels} and max {n_events} events...")
    X_skewed, y_skewed = create_skewed_mnist(X, y, sparsity_levels, n_events)

    print(f"Skewed dataset shape: {X_skewed.shape}")
    return X_skewed, y_skewed, X, y

def generate_is_relevant(label_list, relevant_set):
    return [label in relevant_set for label in label_list]

# temp storage
# def showframe(self, abs_index):
#   im_data = self.data_window.get_data_point(abs_index)
#   im_data = im_data.reshape([128, 128, 3])
#   plt.imshow(im_data, cmap='gray')
#   plt.title(f"Index: {abs_index}")
#   plt.axis('off')