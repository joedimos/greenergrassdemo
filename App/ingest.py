from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from pathlib import Path
from ..config import settings

def ingest_folder(folder: str = "./docs"):
    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
    docs = []
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for p in Path(folder).glob("**/*.txt"):
        loader = TextLoader(str(p), encoding="utf-8")
        raw = loader.load()
        chunks = splitter.split_documents(raw)
        docs.extend(chunks)
    
    # Initialize Qdrant client
    client = QdrantClient(url=settings.qdrant_url)
    
    # Create vector store and add documents
    qdrant = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=settings.qdrant_url,
        collection_name="documents",
        prefer_grpc=False
    )
    
    print(f"✓ Ingested {len(docs)} document chunks")

if __name__ == "__main__":
    ingest_folder()
