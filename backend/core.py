import os
from typing import Any, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv


def run_llm(query: str, context: List[dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    doc_search = PineconeVectorStore.from_existing_index(
        index_name="cb-docs-index", embedding=embeddings
    )
    llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=doc_search.as_retriever(),
    #     return_source_documents=True,
    # )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=doc_search.as_retriever(), return_source_documents=True
    )
    res = qa({"question": query, "chat_history": context})
    print(res)
    return res


if __name__ == "__main__":
    load_dotenv()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    run_llm(query="Summarize different pagination strategies")
