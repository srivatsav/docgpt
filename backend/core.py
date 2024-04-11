import os
from typing import Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    doc_search = PineconeVectorStore.from_existing_index(
        index_name="cb-docs-index", embedding=embeddings
    )
    llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=doc_search.as_retriever(),
        return_source_documents=True,
    )
    res = qa({"query": query})
    print(res)
    return res


if __name__ == "__main__":
    load_dotenv()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    run_llm(query="Summarise the list of items for new UI adoption changes")
