# import streamlit as st
# from utils import write_message
# from agent import generate_response

# # Page Config
# st.set_page_config("Ebert", page_icon=":movie_camera:")

# # Set up Session State
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Hi, I'm the Movie Recommendations Chatbot!  How can I help you?"},
#     ]

# # Submit handler
# def handle_submit(message):
#     """
#     Submit handler:

#     You will modify this method to talk with an LLM and provide
#     context using data from Neo4j.
#     """

#     # Handle the response
#     # Generate response not importing correctly
#     with st.spinner('Thinking...'):
#         # Call the agent
#         response = generate_response(message)
#         write_message('assistant', response)
        


# # Display messages in Session State
# for message in st.session_state.messages:
#     write_message(message['role'], message['content'], save=False)

# # Handle any user input
# if prompt := st.chat_input("What is up?"):
#     # Display user message in chat message container
#     write_message('user', prompt)

#     # Generate a response
#     handle_submit(prompt)
# st.write("Movie Recommendations App!")
# question = st.text_input("Input Your Question Here:")
# st.write(f"Your question is: {question}")
# 
# '--------'

from __future__ import annotations
# import streamlit as st
import logging





from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from neo4j import exceptions
import os
import uvicorn
from dotenv import load_dotenv
from agent import generate_response
# Load environment variables from .env file
load_dotenv()


# Allowed CORS origins
origins = [
    "http://127.0.0.1:8000",  # Alternative localhost address
    "http://localhost:8000",
]


# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from agent import generate_response
from utils import get_messages, get_session_id
from tools.user_id import  generate_userId

app = FastAPI()

# Define the input model for the POST request
class ApiChatPostRequest(BaseModel):
    user_input: str = Field(..., description="The chat message to send")
    user_id: str = Field(..., description='Takes the user ID')

# Define the response model
class ApiChatPostResponse(BaseModel):
    response: str

# POST endpoint for receiving user input and returning AI response
@app.post("/api/chat", response_model=ApiChatPostResponse, tags=["chat"])
async def send_chat_message(body: ApiChatPostRequest, request: Request):
    """
    Send a chat message, get a response, and save both the input and output messages.
    """
    user_input = body.user_input
    global user_id
    user_id = body.user_id 
    # Generate response using agent and tools

    response = generate_response(user_input, request, user_id)
    user = generate_userId(user_input, user_id)
    
    # Return the response
    return { "user":user, "response": response}




# GET endpoint for retrieving all saved messages for a session
@app.get("/api/messages", tags=["chat"])
async def get_all_messages(request: Request):
    """
    Get all the saved messages (conversation history) for the current session.
    """
    session_id = get_session_id(request)
    messages = get_messages(session_id)
    return {"messages": messages}

        


