import pickle

def load_model(model_path):
    """
    Load the trained fraud detection model.

    Args:
        model_path (str): Path to the model file.

    Returns:
        object: Loaded model.
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def predict_fraud(model, preprocessed_input):
    """
    Predict whether a transaction is fraudulent.

    Args:
        model (object): Trained fraud detection model.
        preprocessed_input (np.array): Preprocessed input data.

    Returns:
        bool: True if fraudulent, False otherwise.
    """
    prediction = model.predict(preprocessed_input)
    return prediction[0] == 1  # Returns True for fraud
