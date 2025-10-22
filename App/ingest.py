from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from pathlib import Path
from ..config import settings
def ingest_folder(folder: str = "./docs"):
embeddings = OpenAIEmbeddings()
docs = []
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
for p in Path(folder).glob("**/*.txt"):
loader = TextLoader(str(p), encoding="utf-8")
raw = loader.load()
chunks = splitter.split_documents(raw)
docs.extend(chunks)
6
qdrant = Qdrant(url=settings.qdrant_url, collection_name="documents",
prefer_grpc=False, embeddings=embeddings)
qdrant.add_documents(docs)
if __name__ == "__main__":
ingest_folder()
