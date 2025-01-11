import streamlit as st
from src.data_preprocessing import preprocess_user_input
from src.model_prediction import load_model, predict_fraud
from src.explaination_pipeline import create_retrieval_qa_chain, explain_fraud
from src.vectorization import initialize_vectorstore
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Check your .env file.")
        return

    # Load the fraud detection model
    model_path = "fraud_model.pkl"
    model = load_model(model_path)

    # Initialize the vectorstore and RAG pipeline
    vectorstore_path = "fraud_db"
    vectorstore = initialize_vectorstore(vectorstore_path)
    retrieval_qa_chain = create_retrieval_qa_chain(vectorstore, api_key)

    # Streamlit app interface
    st.title("Fraud Detection Application")
    st.subheader("Analyze suspicious transactions for potential fraud")

    # Input fields for transaction details
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.01)
    transaction_time = st.number_input("Transaction Time (seconds since first transaction)", min_value=0.0, step=1.0)
    other_features = st.text_area("Enter additional features (comma-separated):", placeholder="Feature1,Feature2,...")

    if st.button("Check Transaction"):
        if not other_features.strip():
            st.warning("Please fill in all required fields.")
        else:
            # Parse additional features
            try:
                additional_features = [float(x.strip()) for x in other_features.split(",")]
            except ValueError:
                st.error("Invalid feature values. Please enter numeric values.")
                return

            # Preprocess user input
            preprocessed_input = preprocess_user_input(amount, transaction_time, additional_features)

            # Predict fraud
            is_fraud = predict_fraud(model, preprocessed_input)

            if is_fraud:
                st.error("Transaction flagged as FRAUDULENT!")
                explanation = explain_fraud(
                    retrieval_qa_chain,
                    f"Transaction of ${amount} at {transaction_time} seconds. Features: {other_features}."
                )
                st.write("### Explanation:")
                st.write(explanation)
            else:
                st.success("Transaction is NOT fraudulent.")

if __name__ == "__main__":
    main()