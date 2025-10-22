from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from ..config import settings


def build_rag():
    # Initialize embeddings with API key
    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
    
    # Connect to Qdrant vector store
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=settings.qdrant_url,
        collection_name="documents",
        prefer_grpc=False
    )
    
    retriever = qdrant.as_retriever(search_kwargs={"k": 4})
    
    # Initialize LLM
    llm = OpenAI(temperature=0, api_key=settings.openai_api_key)
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # Include sources in response
    )
    
    return qa

def answer_query(qa_chain, question: str) -> str:
    result = qa_chain.invoke({"query": question})
    return result["result"]
