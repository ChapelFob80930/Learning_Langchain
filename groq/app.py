import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import time

load_dotenv()

## loading api key from env
groq_api_key = os.environ['GROQ_API_KEY']

if "vectorDB" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://www.pcgamer.com/games/fps/destiny-2-just-got-weird-launch-trailer-leans-hard-into-time-travel-and-looks-more-like-a-control-crossover/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200) 
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectorDB = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
    
st.title("ChatGroq using langchain - end to end project")
llm = ChatGroq(groq_api_key=groq_api_key, model_name = "meta-llama/llama-4-scout-17b-16e-instruct")
prompt = ChatPromptTemplate.from_template(
""" 

Answer the questions based on the provided context only.
Please provide the most accurate response based on the context and question. 
<context>
{context}
<context>
Questions: {input}

"""
)

doc_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectorDB.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

prompt = st.text_input("Enter your questions related to destiny 2 edge of fate launch trailer")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    print(f"Your query was processed in {time.process_time() - start}")
    st.write(response['answer'])
    
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------")