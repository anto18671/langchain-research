from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load and split documents
def load_and_split_documents(path_to_docs):
    # Initialize empty list
    documents = []

    # Iterate through directory
    for root, dirs, files in os.walk(path_to_docs):
        for file in files:
            # Build file path
            file_path = os.path.join(root, file)

            # Load markdown or text
            if file.endswith(".md") or file.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())

            elif file.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
                pdf_docs = loader.load()
                documents.extend(pdf_docs)

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Filter valid chunks
    clean_chunks = []
    for chunk in chunks:
        if isinstance(chunk.page_content, str) and chunk.page_content.strip():
            clean_chunks.append(chunk)

    # Return cleaned chunks
    return clean_chunks
