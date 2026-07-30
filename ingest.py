from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from embedding import embed_model

load_dotenv()

CHROMA_PATH = "./chroma_db"

def ingest(path:str):

    data = PDFPlumberLoader(file_path=path)
    documents = data.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = text_splitter.split_documents(documents)

    Chroma.from_documents(documents=chunks, embedding=embed_model, persist_directory=CHROMA_PATH)
    return(len(chunks))