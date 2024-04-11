from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone

documents = []


def ingest_docs() -> None:
    for file in os.listdir("./docs"):
        if file.endswith('.pdf'):
            pdf_path = './docs/' + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    print(f"loaded ${len(documents)} documents")
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"split into {len(texts)} chunks")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = PineconeVectorStore.from_documents(texts, embeddings, index_name="cb-docs-index")
    print("*** created vectors with embeddings ***")
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

if __name__ == "__main__":
    load_dotenv()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    ingest_docs()
