import streamlit as st
from llm import llm, embeddings
from graph import graph
import logging
from retry import retry
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""
index_name = "moviePlots"

vector_store = Neo4jVector.from_existing_index(
    embedding=embeddings,                              # (1)
    graph=graph,                             # (2)
    index_name="moviePlots",                 # (3)
    node_label="Movie",                      # (4)
    # text_node_properties=["plot"],               # (5)
    embedding_node_property="plotEmbedding", # (6)
    retrieval_query=retrieval_query,
)



vector_store = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        graph=graph,
        index_name=index_name,
        node_label="Movie",
        text_node_properties=["plot"],
        embedding_node_property="plotEmbedding",
        retrieval_query=retrieval_query,
    )


if vector_store is None:
    vector_store = Neo4jVector.from_existing_index(
    embedding=embeddings,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    retrieval_query=retrieval_query,
    )
        
retriever = vector_store.as_retriever()

# Put context in a readable format ex:
# context = [{'m.title': 'Leaving Las Vegas'}, {'m.title': 'Piano, The'}, ...]  # Example structure
# formatted_context = "\n".join([f"Title: {movie['m.title']}" for movie in context])

# # Update instructions to include the formatted context
# instructions = (
#     f"Use the given context to answer the question. Context:\n{formatted_context}\n"
#     "If you don't know the answer, say you don't know."

instructions = (
    
    "Use the given context to answer the question. Context: \n{context}\n"
    "If you don't know the answer, say you don't know"

)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

def get_movie_plot(input):
    return plot_retriever.invoke({"input": input})
