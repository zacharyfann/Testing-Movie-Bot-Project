from langchain_core.chat_history import BaseChatMessageHistory
from operator import itemgetter
from typing import List
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uuid


session_data = {}

app = FastAPI()

@app.middleware("http")
def get_session_id(request:Request):
    session_id = request.cookies.get("session_id")
    
    if not session_id:
        session_id = str(uuid.uuid4())  # Generate a new session ID
        session_data[session_id] = {"history": []}  # Initialize session data
    return session_id


# utils.py



# In-memory session storage, can be replaced with database or persistent storage
session_data = {}

def save_message(session_id: str, role: str, content: str, user_id:int):
    """
    Save a message to the session_data under the given session_id.
    Ensures that 'messages' key is properly initialized.
    """
    # Initialize session data if session_id does not exist
    if session_id not in session_data:
        session_data[session_id] = {"messages": []}  # Initialize messages as an empty list

    # Ensure the 'messages' key exists under the session_id
    if "messages" not in session_data[session_id]:
        session_data[session_id]["messages"] = []  # Initialize messages list if not present

    # Append the message to the session's 'messages' list
    session_data[session_id]["messages"].append({"role": role, "content": content, "user_id":user_id})


def get_messages(session_id: str):
    """
    Retrieve all messages from the session state for a given session_id.
    """
    # Return the messages for the session, or an empty list if session_id or messages key doesn't exist
    return session_data.get(session_id, {}).get("messages", [])
