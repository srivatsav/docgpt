from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from typing import List
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

pdf_documents = []
java_documents = []
md_documents = []
csv_documents = []


# Function to get all files recursively with given extensions
def get_files_with_extensions(directory, extensions):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files


# Extensions to consider
pdf_extensions = [".pdf"]
java_extensions = [".java"]
md_extensions = [".md"]
csv_extensions = [".csv"]


def ingest_pdf_docs(files: List[str]) -> None:
    for file in files:
        if not isinstance(file, str):
            continue
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file)
            pdf_documents.extend(loader.load())
    print(f"loaded ${len(pdf_documents)} PDF documents")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(pdf_documents)
    print(f"split into {len(texts)} chunks")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = PineconeVectorStore.from_documents(
        texts, embeddings, index_name="cb-docs-index"
    )
    print("*** created vectors with embeddings ***")
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

def ingest_java_docs(files: List[str]) -> None:
    for file in files:
        if not isinstance(file, str):
            continue
        if file.endswith(".java"):
            loader = GenericLoader.from_filesystem(
                file,
                glob="*",
                suffixes=[".java"],
                parser=LanguageParser(language=Language.JAVA, parser_threshold=1000),
            )
            java_documents.extend(loader.load())
    print(f"loaded ${len(java_documents)} JAVA documents")
    text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.JAVA, chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(java_documents)
    print(f"split into {len(texts)} hunks")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = PineconeVectorStore.from_documents(
        texts, embeddings, index_name="cb-docs-index"
    )
    print("*** created vectors with embeddings ***")
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

def ingest_md_docs(files: List[str]) -> None:
    for file in files:
        if not isinstance(file, str):
            continue
        if file.endswith(".md"):
            loader = TextLoader(file)
            md_documents.extend(loader.load())
    print(f"loaded ${len(md_documents)} MD documents")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(md_documents)
    print(f"split into {len(texts)} chunks")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = PineconeVectorStore.from_documents(
        texts, embeddings, index_name="cb-docs-index"
    )
    print("*** created vectors with embeddings ***")
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


def ingest_csv_docs(files: List[str]) -> None:
    for file in files:
        if not isinstance(file, str):
            continue
        if file.endswith(".csv"):
            loader = CSVLoader(file)
            csv_documents.extend(loader.load())
    print(f"loaded ${len(csv_documents)} CSV documents")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(csv_documents)
    print(f"split into {len(texts)} chunks")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = PineconeVectorStore.from_documents(
        texts, embeddings, index_name="cb-docs-index"
    )
    print("*** created vectors with embeddings ***")
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


if __name__ == "__main__":
    load_dotenv()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    # pdf_files = get_files_with_extensions("./docs", pdf_extensions)
    # java_files = get_files_with_extensions("./docs", java_extensions)
    # md_files = get_files_with_extensions("./docs/release-notes", md_extensions)
    # ingest_pdf_docs(files=pdf_files)
    # ingest_java_docs(files=java_files)
    # ingest_md_docs(files=md_files)
    csv_files = get_files_with_extensions("./docs/csv", csv_extensions)
    ingest_csv_docs(files=csv_files)
