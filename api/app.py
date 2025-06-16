from typing import List, Union
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
load_dotenv()  # Load environment variables from .env file

from fastapi.middleware.cors import CORSMiddleware




# Set all CORS enabled origins

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


app = FastAPI(
    title = "LangChain API",
    version = "1.0",
    description= "API for LangChain with Google Gemini AI and Ollama",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)



@app.get("/")
def read_root():
    return {"message": "Welcome to the LangChain API with Google Gemini AI and Ollama!"}

add_routes(app, ChatGoogleGenerativeAI(model="gemini-1.5-flash",
    temperature=0.2,
    max_tokens=1000,
    top_p=0.9,
    top_k=40,), path="/google")

# add_routes(app, ChatOpenAI(), path="/openai")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
    temperature=0.2,
    max_tokens=1000,
    top_p=0.9,
    top_k=40,)

# model = ChatOpenAI()

llm = Ollama(model="llama2")

prompt1 = prompt = ChatPromptTemplate.from_template("Write me a essay on {topic} in 100 words")
prompt2 = prompt = ChatPromptTemplate.from_template("Write me a essay on {topic} in 100 words")

add_routes(app, prompt1|model|StrOutputParser(), path="/essay")

add_routes(app, prompt2|llm|StrOutputParser(), path="/poem")

if __name__ == "__main__":
    uvicorn.run(app,  host="localhost", port = 8000)
    