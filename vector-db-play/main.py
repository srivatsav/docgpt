from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

if __name__ == "__main__":
    load_dotenv()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    loader = PyPDFLoader("/Users/srivatsav.gorti/Desktop/New UI Adoption Enhancements Blockers Grooming.pdf")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = PineconeVectorStore.from_documents(texts, embeddings, index_name="doc-embedding-index")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    query = "what are the top 2 blockers for new UI adoption? give them to me by priority"
    result = qa({"query": query})
    print(result)
