import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_fraud(model, transaction_data):
    prediction = model.predict([transaction_data])
    return prediction[0]