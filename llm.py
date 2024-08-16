import streamlit as st
from langchain_ollama.llms import OllamaLLM

# Create the LLM
llm = OllamaLLM(model='llama3')

# Create the Embedding model
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="avr/sfr-embedding-mistral")