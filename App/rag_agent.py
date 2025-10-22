from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from ..config import settings
# Simple builder for demo; replace OpenAI with your preferred LLM
def build_rag():
# Ensure OPENAI_API_KEY is available in environment (langchain/OpenAI
will pick it up)
embeddings = OpenAIEmbeddings()
# qdrant client is optional; `Qdrant` vectorstore wrapper will connect
via URL
qdrant = Qdrant(url=settings.qdrant_url, collection_name="documents",
prefer_grpc=False, embeddings=embeddings)
retriever = qdrant.as_retriever(search_kwargs={"k": 4})
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
return qa
def answer_query(qa_chain, question: str) -> str:
return qa_chain.run(question)
