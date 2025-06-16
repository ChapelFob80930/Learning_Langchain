from langchain_google_genai import ChatGoogleGenerativeAI #It is the OpenAI LLM
from langchain_core.prompts import ChatPromptTemplate #It is the prompt template for the chat
from langchain_core.output_parsers import StrOutputParser #It is the output parser for the chat

import streamlit as st 
import os
from dotenv import load_dotenv 

load_dotenv()  # Load environment variables from .env file
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a helpful assistant. Please respond to the user's queries"),
    ("user", "Questions: {question}"),
    ]
)

## Streamlit Framework

st.title("Chatbot with LangChain and Google Gemini AI")
input_text = st.text_input("Ask a question:")

## OpenAI LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_tokens=1000,
    top_p=0.9,
    top_k=40,
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
        