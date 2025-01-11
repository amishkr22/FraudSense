import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
data = pd.read_csv("C:\VILLGAX\CODES\Data\creditcard.csv")

# Normalize 'Amount'
scaler = StandardScaler()
data['NormAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Drop unused columns
data = data.drop(['Amount', 'Time'], axis=1)

# Split features and target
X = data.drop(['Class'], axis=1)
Y = data['Class']
X.shape
# Split data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# Save the trained model
with open("fraud_model.pkl", "wb") as model_file:
    pickle.dump(rfc, model_file)

# Save the scaler
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully.")