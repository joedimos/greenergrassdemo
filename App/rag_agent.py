from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, Optional
import logging
from app.config import settings

# Set up logging
logger = logging.getLogger(__name__)

def build_rag(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    search_k: int = 4
):
    """
    Build a Retrieval-Augmented Generation (RAG) chain using Qdrant + OpenAI
    with LCEL (LangChain Expression Language)
    
    Args:
        model_name: OpenAI model to use
        temperature: Model temperature (0.0 to 1.0)
        search_k: Number of documents to retrieve
        
    Returns:
        RAG chain ready for inference
    """
    try:
        # 1. Initialize embeddings
        embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model="text-embedding-3-small"
        )

        # 2. Connect to existing Qdrant collection
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=settings.qdrant_url,
            collection_name="documents",
            prefer_grpc=False
        )

        # 3. Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": search_k}
        )

        # 4. Define the LLM
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=1000
        )

        # 5. Define the system prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the following context to answer the user's question.
            
Guidelines:
- Answer based ONLY on the provided context
- If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the available documents."
- Be concise but helpful
- If the question is unclear, ask for clarification

Context:
{context}"""),
            ("human", "{question}")
        ])

        # 6. Build RAG chain using LCEL
        rag_chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        logger.info("RAG chain built successfully")
        return rag_chain
        
    except Exception as e:
        logger.error(f"Error building RAG chain: {str(e)}")
        raise

def _format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def answer_query(rag_chain, question: str) -> str:
    """
    Send a query to the RAG chain and return the answer text.
    
    Args:
        rag_chain: The RAG chain instance
        question: User's question
        
    Returns:
        Answer string
    """
    try:
        if not question or not question.strip():
            return "Please provide a valid question."
            
        result = rag_chain.invoke(question)
        return result
        
    except Exception as e:
        logger.error(f"Error processing query '{question}': {str(e)}")
        return "Sorry, I encountered an error while processing your question. Please try again."

# Alternative function that returns more detailed response
def answer_query_detailed(rag_chain, question: str) -> Dict[str, Any]:
    """
    Send a query to the RAG chain and return detailed response including sources.
    
    Args:
        rag_chain: The RAG chain instance
        question: User's question
        
    Returns:
        Dictionary containing answer and metadata
    """
    try:
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
                "error": "Empty question"
            }
        
        # For detailed responses, we need to modify the approach
        # This is a simplified version - you might need to adjust based on your needs
        answer = rag_chain.invoke(question)
        
        return {
            "answer": answer,
            "question": question,
            "sources": []  # You can enhance this to include actual source documents
        }
        
    except Exception as e:
        logger.error(f"Error processing query '{question}': {str(e)}")
        return {
            "answer": "Sorry, I encountered an error while processing your question.",
            "question": question,
            "error": str(e)
        }

# Utility function to test the RAG system
def test_rag_system():
    """Test the RAG system with a sample question"""
    try:
        rag_chain = build_rag()
        test_question = "What is the main topic of the documents?"
        answer = answer_query(rag_chain, test_question)
        print(f"Question: {test_question}")
        print(f"Answer: {answer}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

# If you want to create a singleton instance
_rag_chain_instance = None

def get_rag_chain():
    """Get or create a singleton RAG chain instance"""
    global _rag_chain_instance
    if _rag_chain_instance is None:
        _rag_chain_instance = build_rag()
    return _rag_chain_instance
