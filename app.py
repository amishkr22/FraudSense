import streamlit as st
from src.data_preprocessing import preprocess_user_input
from src.model_prediction import predict_fraud
from src.explaination_pipeline import create_retrieval_qa_chain, explain_fraud
from src.vectorization import initialize_vectorstore
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    st.title("Fraud Detection Application")
    st.subheader("Analyze suspicious transactions for potential fraud")

    # Input fields for transaction details
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.01)
    pca_features = st.text_area(
        "Enter PCA features (comma-separated):",
        placeholder="Feature1,Feature2,..."
    )

    # Initialize ChromaDB vector store and RAG pipeline
    vectorstore_path = "fraud_db"
    vectorstore = initialize_vectorstore(vectorstore_path)
    api_key = os.getenv("OPENAI_API_KEY")
    retrieval_qa_chain = create_retrieval_qa_chain(vectorstore, api_key)

    if st.button("Check Transaction"):
        if not pca_features.strip():
            st.warning("Please fill in all required fields.")
        else:
            # Parse PCA features
            try:
                pca_values = [float(x.strip()) for x in pca_features.split(",")]
                assert len(pca_values) == 28, "PCA features must have exactly 28 values."
            except (ValueError, AssertionError) as e:
                st.error(f"Invalid input: {e}")
                return

            # Preprocess input
            preprocessed_input = preprocess_user_input(amount, pca_values)

            # Predict fraud
            is_fraud = predict_fraud(preprocessed_input)

            if is_fraud:
                st.error("Transaction flagged as FRAUDULENT!")

                # Generate explanation using RAG
                query = f"Transaction of ${amount} with features: {pca_features}."
                explanation = explain_fraud(retrieval_qa_chain, query)
                st.write("### Explanation:")
                st.write(explanation)
            else:
                st.success("Transaction is NOT fraudulent.")

if __name__ == "__main__":
    main()