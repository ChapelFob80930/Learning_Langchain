import requests
import streamlit as st

def get_google_gemini_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke", json={"input": {'topic':input_text}})
    print(response.json())
    return response.json()['output']['content']

def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke", json={"input": {'topic':input_text}})
    print(response.json())
    return response.json()['output']


st.title("LangChain API Client")
input_text = st.text_input("Enter a topic for the essay")
input_text2 = st.text_input("Enter a topic for the poem")



if input_text:
    st.write(get_google_gemini_response(input_text))
    get_google_gemini_response(input_text)
    
if input_text2:
    st.write(get_ollama_response(input_text2))
    get_ollama_response(input_text2)