from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

def initialize_vectorstore(persist_directory):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    return vectorstore

def add_fraud_data_to_vectorstore(vectorstore, fraud_data):
    texts = [" ".join(map(str, row)) for row in fraud_data.drop('Class', axis=1).values]
    vectorstore.add_texts(texts=texts)