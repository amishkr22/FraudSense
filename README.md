# Fraud Detection with Explainable AI and RAG Pipeline

This project is a **fraud detection system** that uses machine learning, Retrieval-Augmented Generation (RAG), and Explainable AI to detect and explain fraudulent transactions. It provides a user-friendly **Streamlit app** for fraud analysis and integrates **Random Forest Classifier** and **LLMs** to offer detailed explanations for flagged transactions.

---

## 🚀 Features
- **Fraud Detection**: Detects fraudulent transactions using a pre-trained Random Forest Classifier.
- **Explainable AI**: Explains flagged fraudulent transactions using Retrieval-Augmented Generation (RAG).
- **RAG Pipeline**:
  - Uses **ChromaDB** for storing and retrieving past fraudulent transactions.
  - Generates human-readable explanations via pre-trained language models (e.g., OpenAI's GPT-4).
- **Streamlit App**: Provides an intuitive interface for users to input transaction details and view results.

---

## 📂 Project Structure Should be like this
```
fraud_detection_app/
├── app.py                     # Main Streamlit application
├── train_model.py             # Script to train the Random Forest model
├── creditcard.csv             # Dataset for training and testing
├── fraud_model.pkl            # Trained Random Forest model
├── scaler.pkl                 # Scaler for preprocessing transaction amounts
├── src/
│   ├── data_preprocessing.py  # Preprocessing logic for user inputs
│   ├── model_prediction.py    # Model loading and prediction
│   ├── explaination_pipeline.py # RAG pipeline for generating explanations
│   ├── vectorization.py       # Vector store setup and initialization
```

---

## 📊 Dataset
The project uses the **Kaggle Credit Card Fraud Detection Dataset**, which contains:
- **Transactions**: 284,807 samples.
- **Features**: 
  - `Time`: Time since the first transaction.
  - `Amount`: Transaction amount.
  - `V1` to `V28`: PCA-transformed features.
- **Target**: `Class` (1 for fraud, 0 for valid).

---

## 🔧 Installation
### Prerequisites
1. **Python 3.8+**
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Files to Prepare
- **`creditcard.csv`**: Dataset file in the project directory.
- **`fraud_model.pkl`**: Trained Random Forest model.
- **`scaler.pkl`**: Pretrained scaler for amount normalization.

---

## 🖥️ Usage
### 1. Train the Model
To train the Random Forest Classifier:
```bash
python train_model.py
```

### 2. Run the Streamlit App
Start the Streamlit app to analyze transactions:
```bash
streamlit run app.py
```

### 3. Input Details
Provide the following details in the Streamlit app:
- **Transaction Amount** (e.g., `$1000`).
- **PCA Features** (comma-separated values for `V1` to `V28`).

---

## 🛠️ Key Components
### 1. Fraud Detection
- A **Random Forest Classifier** predicts whether a transaction is fraudulent based on the input features.

### 2. RAG Pipeline
- **ChromaDB** stores embeddings of past fraudulent transactions.
- **LLMs (e.g., GPT-4)** generate explanations for flagged transactions.

### 3. Explanation Example
If a transaction is flagged as fraudulent, the system provides:
```
Transaction flagged as FRAUDULENT!
Explanation:
This transaction resembles past fraudulent cases due to high transaction amount and PCA patterns.
Similar cases:
- Transaction of $5000 flagged as fraud.
- Transaction of $4800 flagged as fraud.
```

---

## 🧪 Testing
Use the `test_app.py` script to test the app:
```bash
python test_app.py
```

Sample inputs:
- **Non-Fraudulent Transaction**:
  - Amount: `$100`
  - PCA Features: `-1.359, 1.191, -0.206, ..., 0.207`
- **Fraudulent Transaction**:
  - Amount: `$5000`
  - PCA Features: `2.536, -1.054, 1.967, ..., -0.185`

---

## 📚 Technologies Used
- **Machine Learning**: Random Forest Classifier for fraud detection.
- **LLMs**: GPT-based models for generating explanations.
- **ChromaDB**: Vector store for RAG pipeline.
- **Streamlit**: Interactive app interface.
- **Python Libraries**:
  - `sklearn`
  - `langchain`
  - `streamlit`
  - `pandas`
  - `numpy`

---
