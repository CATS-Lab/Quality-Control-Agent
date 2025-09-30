import os
import argparse
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).parent
RAG_DIR = BASE_DIR / "RAG"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse RAG input arguments.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the input file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    rag_path = RAG_DIR / args.path
    loader = PyPDFLoader(str(rag_path))
    all_pages = loader.load()
    pages = []
    for i, doc in enumerate(all_pages):
        pages.append(doc)

    print("Splitting documents ...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    print(f"Total chunks: {len(docs)}")

    print("Initializing embeddings ...")
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

    print("Building FAISS vectorstore ...")
    vectordb = FAISS.from_documents(docs, embedding_model)

    print("Saving FAISS vectorstore locally ...")
    vectordb.save_local(RAG_DIR / "vector_db")
    print("Done!")
