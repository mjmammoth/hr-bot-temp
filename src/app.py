from dotenv import load_dotenv
import streamlit as st

import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from typing import List, Tuple
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document

langchain.verbose = True
load_dotenv()


st.set_page_config(page_title="Cloudsmiths HR Bot")
st.header("Ask HR a question 💬")


CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="pgvector",
    port=5432,
    database="postgres",
    user="postgres",
    password="postgres",
)

with st.spinner("Loading..."):
    embeddings = OpenAIEmbeddings()
    db = PGVector(connection_string=CONNECTION_STRING, collection_name="hr_policy", embedding_function=embeddings)

user_question = st.text_input("Ask the Cloudsmiths HR bot a question about an implemented policy:")

if user_question:
    docs: List[Tuple[Document, float]] = db.similarity_search(user_question)
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
    st.write(response)
