import os

from dotenv import load_dotenv

import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader

langchain.verbose = False
load_dotenv()

documents = []
# for each file in the training-data folder, load the text and add it to the documents
for filename in os.listdir('../training-data'):
    loader = TextLoader('../training-data/' + filename)
    documents += loader.load()

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=630, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
# for doc in docs:
#     print(doc)
#     print('\n')
#
# print(len(docs))
#
# exit()
embeddings = OpenAIEmbeddings()

CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host="localhost",
        port=5433,
        database="postgres",
        user="postgres",
        password="postgres",
)


with get_openai_callback() as cb:
    db = PGVector.from_documents(
            embedding=embeddings,
            documents=docs,
            connection_string=CONNECTION_STRING,
            collection_name="hr_policy",
    )
    print(cb)
