import streamlit as st
from src.data_processing import load_and_preprocess_data
from src.model_prediction import load_model, predict_fraud
from src.explanation_pipeline import create_retrieval_qa_chain, explain_fraud
from src.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Load pre-trained model
    model_path = "fraud_model.pkl"
    model = load_model(model_path)

    # Initialize vectorstore for retrieval
    vectorstore = Chroma(persist_directory="fraud_db", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
    retrieval_qa_chain = create_retrieval_qa_chain(vectorstore, api_key)

    # Streamlit app interface
    st.title("Fraud Detection Application")

    # User input for transaction details
    amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    transaction_time = st.text_input("Transaction Time (e.g., 2 AM)")
    device_info = st.text_input("Device Information")

    if st.button("Check Transaction"):
        transaction_data = [amount, transaction_time, device_info]  # Simplified example
        is_fraud = predict_fraud(model, transaction_data)

        if is_fraud:
            st.error("Transaction flagged as FRAUDULENT!")
            explanation = explain_fraud(retrieval_qa_chain, f"Transaction of ${amount} at {transaction_time} from {device_info}.")
            st.write("Explanation:", explanation)
        else:
            st.success("Transaction is NOT fraudulent.")

if __name__ == "__main__":
    main()