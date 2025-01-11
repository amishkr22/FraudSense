import numpy as np
import pickle

# Load the scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

def preprocess_user_input(amount, time, pca_features):
    """
    Preprocess user input to match the model's expected input format.

    Args:
        amount (float): Transaction amount.
        time (float): Transaction time.
        pca_features (list): PCA-transformed features (V1 to V28).

    Returns:
        np.array: Preprocessed input ready for the model.
    """
    # Normalize the amount
    norm_amount = scaler.transform([[amount]])[0, 0]

    # Combine all features
    input_features = [norm_amount] + pca_features
    return np.array(input_features).reshape(1, -1)
