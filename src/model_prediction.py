import pickle

# Load the trained model
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

def predict_fraud(preprocessed_input):
    """
    Predict whether a transaction is fraudulent.

    Args:
        preprocessed_input (np.array): Preprocessed input data.

    Returns:
        bool: True if fraudulent, False otherwise.
    """
    prediction = model.predict(preprocessed_input)
    return prediction[0] == 1  # Returns True for fraud