from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def create_retrieval_qa_chain(vectorstore, api_key):
    retriever = vectorstore.as_retriever()
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        A transaction was flagged as suspicious:
        {question}

        Relevant past transactions and patterns:
        {context}

        Based on the provided details, explain why this transaction might be fraudulent.
        """
    )
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=OpenAI(api_key=api_key),
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return retrieval_qa