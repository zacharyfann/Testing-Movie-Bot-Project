from tokenize import maybe
from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
# from utils import get_session_id
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from fastapi import FastAPI
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from typing import List, Dict
import streamlit as st
# Initialize FastAPI app
app = FastAPI()


# FastAPI model for incoming request
class ApiChatPostRequest(BaseModel):
    user_input: str = Field(..., description="The chat message to send")
    user_id: str

from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ClientError

HOST = st.secrets["NEO4J_URI"]
PASSWORD = st.secrets["NEO4J_PASSWORD"]
USER = st.secrets["NEO4J_USERNAME"]
DATABASE = st.secrets["NEO4J_DATABASE"]
global user_input, user_id
def generate_userId(user_input, user_id):
        
        user_id = user_id
        user_input = user_input
        return {"userId":user_id}

def execute_query(query, params):
    
        with GraphDatabase.driver(
            HOST, auth=basic_auth(USER, PASSWORD), database=DATABASE
        ) as driver:
            return driver.execute_query(query, params)
        query = """
        MATCH (u:User {userId: $user_id})-[r:RATED]->(m:Movie)
        RETURN m.title AS movie_name, r.rating AS rating
        """
        params = {user_id: user_id}
        
        answer = execute_query(query, params)
        
        return {"message":answer}


CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer at generating Cypher to provide relevant recommendations based on the ratings from a user with the id of userId.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Fine Tuning:

For movie titles that begin with "The", move "the" to the end. For example "The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".
Remember RETURN can only be used at the end of a query!
When using a parameter userId, make sure that the id is in quotes/ is a string value!                                        


Order of cypher calls:

1. To find movies that a specific user has rated:
```

:param user_id -> {userId}
MATCH (u:User {{userId: userId}})-[r:RATED]->(m:Movie)
RETURN m.title AS movie_title, r.rating AS user_rating
```

Based on the results of 1, find a movie that they would like using 2.

2. To find movies similar to the specified movie:
```
MATCH(m:Movie {{title:$movie_name}})

CALL db.index.vector.queryNodes('moviePlots', 10, m.plotEmbedding)
YIELD node, score

RETURN node.title AS title, node.plot AS plot, score
```

Schema:
{schema}

Question:
{question}

userId:
{userId}

"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

review_chat = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    validate_cypher=True,
    cypher_prompt=cypher_prompt
)

# keyerror:schema
# change the userId and make it a working parameter so the llm can actually reference it